import asyncio
import itertools
import logging
import sys
import threading
import time
from abc import ABC, abstractmethod

from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    stop_after_attempt,
    wait_exponential,
)

from ..manifest import Manifest

log = logging.getLogger(__name__)


async def _with_retry(
    coro_fn,
    *args,
    max_attempts: int = 20,
) -> None:
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, max=60),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    ):
        with attempt:
            await coro_fn(*args)


class BaseWatcher(ABC):
    @abstractmethod
    async def _catch_up(
        self,
        queue: asyncio.Queue,
        manifest: "Manifest",
    ) -> str: ...

    @abstractmethod
    async def _watch(
        self,
        queue: asyncio.Queue,
        manifest: "Manifest",
        catchup_done: asyncio.Event,
    ) -> None: ...

    async def run(
        self,
        queue: asyncio.Queue,
        manifest: "Manifest",
        catchup_done: asyncio.Event | None = None,
    ) -> None:
        watch_ready = asyncio.Event()
        watch_task = asyncio.create_task(self._watch(queue, manifest, watch_ready))
        try:
            catchup_message = await self._catch_up(queue, manifest)

            stop_spinning = threading.Event()

            def _spin() -> None:
                frames = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
                while not stop_spinning.is_set():
                    sys.stderr.write(f"\r{next(frames)} Catch-up in progress …  ")
                    sys.stderr.flush()
                    time.sleep(0.1)

            spin_thread = threading.Thread(target=_spin, daemon=True) if sys.stderr.isatty() else None
            if spin_thread is not None:
                spin_thread.start()
            try:
                await queue.join()
            finally:
                if spin_thread is not None:
                    stop_spinning.set()
                    spin_thread.join()
                    sys.stderr.write("\r" + " " * 40 + "\r")
                    sys.stderr.flush()

            log.info(catchup_message)
            watch_ready.set()
            if catchup_done:
                catchup_done.set()
            await watch_task
        except Exception:
            watch_task.cancel()
            raise


class Monitor:
    def __init__(self, queue: asyncio.Queue, manifest: "Manifest"):
        self._watchers: list[BaseWatcher] = []
        self._queue = queue
        self._manifest = manifest

    def add_watcher(self, watcher: BaseWatcher) -> None:
        self._watchers.append(watcher)

    async def run(self, startup_done: asyncio.Event | None = None) -> None:
        watcher_events = [asyncio.Event() for _ in self._watchers]
        tasks = [
            asyncio.create_task(
                _with_retry(watcher.run, self._queue, self._manifest, event)
            )
            for watcher, event in zip(self._watchers, watcher_events)
        ]
        await asyncio.gather(*[event.wait() for event in watcher_events])
        if startup_done:
            startup_done.set()
        await asyncio.gather(*tasks)
