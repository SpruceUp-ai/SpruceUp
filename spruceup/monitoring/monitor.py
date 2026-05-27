import asyncio
import logging
from abc import ABC, abstractmethod

from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    stop_after_attempt,
    wait_exponential,
)

from ..manifest import Manifest
from ..models import SyncTask

log = logging.getLogger(__name__)


async def _with_retry(
    coro_fn,
    *args,
    max_attempts: int = 20,
) -> None:
    """
    Retry coro_fn(*args) on failure with exponential backoff.

    Wrap `watcher.run()` so a transient crash doesn't kill the watcher.
    Each failed attempt is logged.
    Retries up to `max_attempts` times before giving up.
    The final exception is re-raised so `Monitor.run()` can surface it.

    Note: this wraps the starting of a watcher.
    A healthy watcher's own `run()` loop never returns, so it never re-enters
    this retry logic.
    """
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, max=60),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
        # retry=retry_if_exception_type((OSError, ConnectionError)) # can be expanded to retry for specific transient exceptions and permanently fail for others
    ):
        with attempt:
            await coro_fn(*args)


class _BufferedQueue:
    """
    Buffers live watch events until catch-up is complete.

    Because LocalFileWatcher starts watching before catch-up is complete,
    live events may be enqueued before stale versions are processed.
    This would result in out-of-order syncing, where the stale version
    overwrites the latest version.
    The buffer holds live events back so the catch-up versions are processed first.

    Note: the buffer is unbounded and may grow indefinitely. Catch-up is
    typically fast, but a very slow catch-up over a very active directory
    would grow the buffer indefinitely.
    """

    def __init__(self, target: asyncio.Queue):
        self._target = target
        self._buffer: list = []
        self._live = False

    async def put(self, item) -> None:
        if self._live:
            await self._target.put(item)
        else:
            self._buffer.append(item)

    async def flush(self) -> None:
        self._live = True
        for item in self._buffer:
            await self._target.put(item)
        self._buffer.clear()


class BaseWatcher(ABC):
    @property
    @abstractmethod
    def source_type(self) -> str: ...

    @abstractmethod
    async def _catch_up(
        self,
        queue: asyncio.Queue,
        manifest: "Manifest",
        force_reindex: bool = False,
    ) -> None: ...

    @abstractmethod
    async def _watch(
        self,
        queue: "_BufferedQueue",
        manifest: "Manifest",
    ) -> None: ...

    async def run(
        self,
        queue: asyncio.Queue,
        manifest: "Manifest",
        force_reindex: bool = False,
        catchup_done: asyncio.Event | None = None,
    ) -> None:
        buf = _BufferedQueue(queue)
        watch_task = asyncio.create_task(self._watch(buf, manifest))
        try:
            await self._catch_up(queue, manifest, force_reindex)
            await buf.flush()
            if catchup_done:
                catchup_done.set()
            await watch_task
        except Exception:
            watch_task.cancel()
            raise


class Monitor:
    def __init__(
        self,
        queue: asyncio.Queue,
        manifest: "Manifest",
        transform_hash: bytes | None = None,
    ):
        self._watchers: list[BaseWatcher] = []
        self._queue = queue
        self._manifest = manifest
        self._transform_hash = transform_hash

    def add_watcher(self, watcher: BaseWatcher) -> None:
        self._watchers.append(watcher)

    async def run(
        self, force_reindex: bool = False, startup_done: asyncio.Event | None = None
    ) -> None:
        watcher_events = [asyncio.Event() for _ in self._watchers]
        tasks = [
            asyncio.create_task(
                _with_retry(
                    watcher.run, self._queue, self._manifest, force_reindex, event
                )
            )
            for watcher, event in zip(self._watchers, watcher_events)
        ]
        await asyncio.gather(*[event.wait() for event in watcher_events])
        if force_reindex and self._manifest and self._transform_hash is not None:
            self._manifest.update_transform_hash(self._transform_hash)
        if startup_done:
            startup_done.set()
        await asyncio.gather(*tasks)
