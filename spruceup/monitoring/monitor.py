import asyncio
import logging
import pathlib
from abc import ABC, abstractmethod

from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    stop_after_attempt,
    wait_exponential,
)
from watchfiles import awatch, Change
from ..utils.hashing import hash_file_content
from ..models import SyncTask

from ..manifest import Manifest

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
    """during catch-up phase, directs calls to `put` to an internal buffer until flush() is called, then goes live."""

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
    async def run(
        self,
        queue: asyncio.Queue,
        manifest: "Manifest",
        force_reindex: bool = False,
        catchup_done: asyncio.Event | None = None,
    ) -> None: ...


class Monitor:
    def __init__(self, queue: asyncio.Queue, manifest: "Manifest", transform_hash: bytes | None = None):
        self._watchers: list[BaseWatcher] = []
        self._queue = queue
        self._manifest = manifest
        self._transform_hash = transform_hash

    def add_watcher(self, watcher: BaseWatcher) -> None:
        self._watchers.append(watcher)

    async def run(self, force_reindex: bool = False, startup_done: asyncio.Event | None = None) -> None:
        """fire all watchers concurrently → wait until they've all finished
        startup → tell main.py we're live → then just keep the watcher tasks
        alive forever."""
        watcher_events = [asyncio.Event() for _ in self._watchers]
        tasks = [
            asyncio.create_task(
                _with_retry(watcher.run, self._queue, self._manifest, force_reindex, event)
            )
            for watcher, event in zip(self._watchers, watcher_events)
        ]
        await asyncio.gather(*[event.wait() for event in watcher_events])
        if force_reindex and self._manifest and self._transform_hash is not None:
            self._manifest.update_transform_hash(self._transform_hash)
        if startup_done:
            startup_done.set()
        await asyncio.gather(*tasks)


class LocalFileWatcher(BaseWatcher):
    def __init__(self, dir_path: str, data_source_id: int, source_type: str):
        self._root_path = dir_path
        self._data_source_id = data_source_id
        self._source_type = source_type

    @property
    def source_type(self) -> str:
        return self._source_type

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

    async def _catch_up(
        self,
        queue: asyncio.Queue,
        manifest: "Manifest",
        force_reindex: bool = False,
    ) -> None:
        log.info("Scanning %s …", self._root_path)
        con = manifest.connect()
        n_upserts = n_moves = n_deletes = 0
        try:
            cur = con.cursor()
            seen_inodes = set()

            for path in pathlib.Path(self._root_path).rglob("*"):
                if not path.is_file():
                    continue
                inode = path.stat().st_ino
                current_path_str = str(path)
                seen_inodes.add(inode)
                if force_reindex:
                    await queue.put(SyncTask(self._source_type, current_path_str, "upsert", data_source_id=self._data_source_id))
                    n_upserts += 1
                else:
                    row = cur.execute(
                        "SELECT file_path, content_hash FROM files WHERE inode = ? AND data_source_id = ?",
                        (inode, self._data_source_id),
                    ).fetchone()
                    if row is None:
                        await queue.put(SyncTask(self._source_type, current_path_str, "upsert", data_source_id=self._data_source_id))
                        n_upserts += 1
                    else:
                        db_path_val, db_hash = row[0], row[1]
                        if db_hash != hash_file_content(path):
                            await queue.put(SyncTask(self._source_type, current_path_str, "upsert", data_source_id=self._data_source_id))
                            n_upserts += 1
                        elif db_path_val != current_path_str:
                            await queue.put(SyncTask(self._source_type, current_path_str, "move", old_identifier=db_path_val, data_source_id=self._data_source_id))
                            n_moves += 1

            for db_inode, db_path_val in cur.execute(
                "SELECT inode, file_path FROM files WHERE data_source_id = ?",
                (self._data_source_id,),
            ).fetchall():
                if db_inode not in seen_inodes:
                    await queue.put(SyncTask(self._source_type, db_path_val, "delete", data_source_id=self._data_source_id))
                    n_deletes += 1

            log.info(
                "Catch-up complete — %d upsert(s)  %d move(s)  %d delete(s)",
                n_upserts, n_moves, n_deletes,
            )
        finally:
            con.close()

    async def _watch(self, queue: _BufferedQueue, manifest: "Manifest") -> None:
        """
        Long-running process that observes local files in the watched directory for changes.
        Changes are queued for processing by the `Monitor`.
        """
        con = manifest.connect()
        try:
            async for changes in awatch(self._root_path):
                deleted_paths  = {path for change_type, path in changes if change_type == Change.deleted}
                added_paths    = {path for change_type, path in changes if change_type == Change.added}
                modified_paths = {path for change_type, path in changes if change_type == Change.modified}

                added_by_inode = {
                    pathlib.Path(path).stat().st_ino: path
                    for path in added_paths
                    if pathlib.Path(path).exists()
                }

                moves = []
                for old_path in deleted_paths:
                    row = con.execute(
                        "SELECT inode FROM files WHERE file_path = ? AND data_source_id = ?",
                        (old_path, self._data_source_id),
                    ).fetchone()
                    if row and row[0] in added_by_inode:
                        moves.append((old_path, added_by_inode[row[0]]))

                moved_old = {old for old, _ in moves}
                moved_new = {new for _, new in moves}

                n_upserts = n_moves = n_deletes = 0

                for old_path, new_path in moves:
                    await queue.put(SyncTask(self._source_type, new_path, "move", old_identifier=old_path, data_source_id=self._data_source_id))
                    n_moves += 1

                for path in deleted_paths - moved_old:
                    await queue.put(SyncTask(self._source_type, path, "delete", data_source_id=self._data_source_id))
                    n_deletes += 1

                for path in (added_paths - moved_new) | modified_paths:
                    if pathlib.Path(path).is_file():
                        await queue.put(SyncTask(self._source_type, path, "upsert", data_source_id=self._data_source_id))
                        n_upserts += 1

                log.info(
                    "Change detected — %d upsert(s)  %d move(s)  %d delete(s)",
                    n_upserts, n_moves, n_deletes,
                )
        finally:
            con.close()
