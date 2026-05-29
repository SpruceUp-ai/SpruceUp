import asyncio
import logging
import pathlib

from watchfiles import awatch, Change

from ..utils.hashing import hash_file_content, hash_source_ref
from ..models import SyncTask
from ..manifest import Manifest
from .monitor import BaseWatcher

log = logging.getLogger(__name__)


class LocalFileWatcher(BaseWatcher):
    def __init__(self, dir_path: str, data_source_id: int, source_type: str, is_supported):
        self._root_path = str(pathlib.Path(dir_path).resolve())
        self._data_source_id = data_source_id
        self._source_type = source_type
        self._is_supported = is_supported

    @property
    def source_type(self) -> str:
        return self._source_type

    async def _catch_up(
        self,
        queue: asyncio.Queue,
        manifest: "Manifest",
        force_reindex: bool = False,
    ) -> None:
        log.info("Scanning %s …", self._root_path)
        con = manifest.connect()
        n_upserts = n_moves = n_deletes = 0
        if not force_reindex:
            file_records = manifest.get_files_with_metadata(con, self._data_source_id)
            by_inode: dict[int, dict] = {
                int(rec["metadata"]["inode"]): rec
                for rec in file_records
                if "inode" in rec["metadata"]
            }
        else:
            by_inode = {}

        seen_inodes: set[int] = set()

        for path in pathlib.Path(self._root_path).rglob("*"):
            if not path.is_file() or not self._is_supported(str(path)):
                continue
            inode = path.stat().st_ino
            current_path_str = str(path)
            seen_inodes.add(inode)

            if force_reindex:
                await queue.put(SyncTask(self._source_type, current_path_str, "upsert", data_source_id=self._data_source_id))
                n_upserts += 1
            else:
                db_record = by_inode.get(inode)
                if db_record is None:
                    await queue.put(SyncTask(self._source_type, current_path_str, "upsert", data_source_id=self._data_source_id))
                    n_upserts += 1
                else:
                    db_path_val = db_record["source_ref"]
                    db_hash = db_record["content_hash"]
                    if db_hash != hash_file_content(path):
                        await queue.put(SyncTask(self._source_type, current_path_str, "upsert", data_source_id=self._data_source_id))
                        n_upserts += 1
                    elif db_path_val != current_path_str:
                        await queue.put(SyncTask(self._source_type, current_path_str, "move", old_identifier=db_path_val, data_source_id=self._data_source_id))
                        n_moves += 1

        for inode, rec in by_inode.items():
            if inode not in seen_inodes:
                await queue.put(SyncTask(self._source_type, rec["source_ref"], "delete", data_source_id=self._data_source_id))
                n_deletes += 1

        log.info(
            "Catch-up complete — %d upsert(s)  %d move(s)  %d delete(s)",
            n_upserts, n_moves, n_deletes,
        )

    async def _watch(self, queue: asyncio.Queue, manifest: "Manifest", catchup_done: asyncio.Event) -> None:
        buffer: list[SyncTask] = []
        con = manifest.connect()
        async for changes in awatch(self._root_path):
            deleted_paths  = {path for change_type, path in changes if change_type == Change.deleted}
            added_paths    = {path for change_type, path in changes if change_type == Change.added}
            modified_paths = {path for change_type, path in changes if change_type == Change.modified}

            added_by_inode: dict[int, str] = {
                pathlib.Path(path).stat().st_ino: path
                for path in added_paths
                if pathlib.Path(path).exists()
            }

            moves = []
            for old_path in deleted_paths:
                file_id = hash_source_ref(old_path)
                meta = manifest.get_file_metadata(con, file_id)
                inode = int(meta["inode"]) if meta and "inode" in meta else None
                if inode is not None and inode in added_by_inode:
                    moves.append((old_path, added_by_inode[inode]))

            moved_old = {old for old, _ in moves}
            moved_new = {new for _, new in moves}

            for old_path, new_path in moves:
                buffer.append(SyncTask(self._source_type, new_path, "move", old_identifier=old_path, data_source_id=self._data_source_id))
            for path in deleted_paths - moved_old:
                buffer.append(SyncTask(self._source_type, path, "delete", data_source_id=self._data_source_id))
            for path in (added_paths - moved_new) | modified_paths:
                if pathlib.Path(path).is_file() and self._is_supported(path):
                    buffer.append(SyncTask(self._source_type, path, "upsert", data_source_id=self._data_source_id))

            if catchup_done.is_set():
                for task in buffer:
                    await queue.put(task)
                n_upserts = sum(1 for t in buffer if t.change_type == "upsert")
                n_moves   = sum(1 for t in buffer if t.change_type == "move")
                n_deletes = sum(1 for t in buffer if t.change_type == "delete")
                buffer.clear()
                log.info(
                    "Change detected — %d upsert(s)  %d move(s)  %d delete(s)",
                    n_upserts, n_moves, n_deletes,
                )
