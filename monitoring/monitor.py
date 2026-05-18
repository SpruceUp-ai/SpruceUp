import asyncio
import pathlib
import hashlib
import sqlite3
from abc import ABC, abstractmethod

from watchfiles import awatch, Change
from .tasks import SyncTask


def _hash_file(p: pathlib.Path) -> bytes:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.digest()


async def _with_retry(
    coro_fn,
    *args,
    backoff_base: float = 1.0,
    max_backoff: float = 60.0,
) -> None:
    attempt = 0
    while True:
        try:
            await coro_fn(*args)
            return
        except Exception:
            delay = min(backoff_base * (2 ** attempt), max_backoff)
            attempt += 1
            await asyncio.sleep(delay)


class _BufferedQueue:
    """Routes puts to an internal buffer until flush() is called, then goes live."""

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
    @abstractmethod
    async def run(
        self,
        queue: asyncio.Queue,
        db_path: str,
        sync_engine,
        force_reindex: bool = False,
        startup_done: asyncio.Event | None = None,
    ) -> None: ...


class Monitor:
    def __init__(self, queue: asyncio.Queue, db_path: str, sync_engine):
        self._watchers: list[BaseWatcher] = []
        self._queue = queue
        self._db_path = db_path
        self._sync_engine = sync_engine

    def add_watcher(self, watcher: BaseWatcher) -> None:
        self._watchers.append(watcher)

    async def run(self, force_reindex: bool = False, startup_done: asyncio.Event | None = None) -> None:
        watcher_events = [asyncio.Event() for _ in self._watchers]
        tasks = [
            asyncio.create_task(
                _with_retry(w.run, self._queue, self._db_path, self._sync_engine, force_reindex, e)
            )
            for w, e in zip(self._watchers, watcher_events)
        ]
        await asyncio.gather(*[e.wait() for e in watcher_events])
        if startup_done:
            startup_done.set()
        await asyncio.gather(*tasks)


class LocalFileWatcher(BaseWatcher):
    def __init__(self, dir_path: str):
        self._root_path = dir_path

    async def run(
        self,
        queue: asyncio.Queue,
        db_path: str,
        sync_engine,
        force_reindex: bool = False,
        startup_done: asyncio.Event | None = None,
    ) -> None:
        buf = _BufferedQueue(queue)
        watch_task = asyncio.create_task(self._watch(buf, db_path, sync_engine))
        try:
            await self._catch_up(queue, db_path, sync_engine, force_reindex)
            await buf.flush()
            if startup_done:
                startup_done.set()
            await watch_task
        except Exception:
            watch_task.cancel()
            raise

    async def _catch_up(
        self, queue: asyncio.Queue, db_path: str, sync_engine, force_reindex: bool = False
    ) -> None:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        seen_inodes = set()

        for p in pathlib.Path(self._root_path).rglob("*"):
            if not p.is_file():
                continue
            inode = p.stat().st_ino
            current_path = str(p)
            seen_inodes.add(inode)
            if force_reindex:
                await queue.put(SyncTask("local", current_path, "upsert"))
            else:
                row = cur.execute(
                    "SELECT file_path, hash_value FROM files WHERE inode = ?", (inode,)
                ).fetchone()
                if row is None:
                    await queue.put(SyncTask("local", current_path, "upsert"))
                else:
                    db_path_val, db_hash = row[0], row[1]
                    if db_hash != _hash_file(p):
                        await queue.put(SyncTask("local", current_path, "upsert"))
                    elif db_path_val != current_path:
                        await sync_engine.move_file(db_path_val, current_path)

        for db_inode, db_path_val in cur.execute(
            "SELECT inode, file_path FROM files"
        ).fetchall():
            if db_inode not in seen_inodes:
                await sync_engine.delete_file(db_path_val)

        con.close()

    async def _watch(self, queue: _BufferedQueue, db_path: str, sync_engine) -> None:
        con = sqlite3.connect(db_path)
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
                row = con.execute("SELECT inode FROM files WHERE file_path = ?", (old_path,)).fetchone()
                if row and row[0] in added_by_inode:
                    moves.append((old_path, added_by_inode[row[0]]))

            moved_old = {old for old, _ in moves}
            moved_new = {new for _, new in moves}

            for old_path, new_path in moves:
                await sync_engine.move_file(old_path, new_path)

            for path in deleted_paths - moved_old:
                await sync_engine.delete_file(path)

            for path in (added_paths - moved_new) | modified_paths:
                if pathlib.Path(path).exists():
                    await queue.put(SyncTask("local", path, "upsert"))
        con.close()
