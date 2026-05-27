import asyncio
import logging
import pathlib

from watchfiles import awatch, Change

from ..utils.hashing import hash_file_content
from ..models import SyncTask
from ..manifest import Manifest
from .monitor import BaseWatcher, _BufferedQueue

log = logging.getLogger(__name__)


class LocalFileWatcher(BaseWatcher):
    def __init__(self, dir_path: str, data_source_id: int, source_type: str):
        self._root_path = str(pathlib.Path(dir_path).resolve())
        self._data_source_id = data_source_id
        self._source_type = source_type

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
