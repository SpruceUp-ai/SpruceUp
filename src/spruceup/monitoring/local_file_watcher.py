import asyncio
import logging
import pathlib

from watchfiles import awatch, Change

from ..models import SyncTask
from ..manifest import Manifest
from ..connectors.sources.local import make_file_id, file_id_to_inode, file_id_to_path
from .monitor import BaseWatcher

log = logging.getLogger(__name__)


class LocalFileWatcher(BaseWatcher):
    def __init__(self, dir_path: str, data_source_id: int, is_supported):
        self._root_path = str(pathlib.Path(dir_path).resolve())
        self._data_source_id = data_source_id
        self._is_supported = is_supported
        self._known_file_ids: set[str] = set()

    async def _catch_up(
        self,
        queue: asyncio.Queue,
        manifest: "Manifest",
    ) -> str:
        n_upserts = n_deletes = 0

        by_inode: dict[int, tuple[str, float | None]] = {}
        for rec in manifest.get_files_for_source(self._data_source_id):
            fid = rec["file_id"]
            try:
                by_inode[file_id_to_inode(fid)] = (fid, rec["modified_at"])
            except (ValueError, IndexError):
                pass

        seen_inodes: set[int] = set()
        n_skipped = 0

        for path in pathlib.Path(self._root_path).rglob("*"):
            if not path.is_file():
                continue
            if not self._is_supported(str(path)):
                n_skipped += 1
                continue
            stat = path.stat()
            inode = stat.st_ino
            path_str = str(path)
            new_file_id = make_file_id(inode, path_str)
            seen_inodes.add(inode)
            self._known_file_ids.add(new_file_id)

            stored = by_inode.get(inode)
            if stored is None:
                await queue.put(SyncTask(
                    "upsert",
                    current_file_id=new_file_id,
                    data_source_id=self._data_source_id,
                ))
                n_upserts += 1
            else:
                stored_file_id, stored_mtime = stored
                renamed = stored_file_id != new_file_id
                if renamed:
                    await queue.put(SyncTask(
                        "delete",
                        current_file_id=stored_file_id,
                        data_source_id=self._data_source_id,
                    ))
                    n_deletes += 1
                    await queue.put(SyncTask(
                        "upsert",
                        current_file_id=new_file_id,
                        data_source_id=self._data_source_id,
                    ))
                    n_upserts += 1
                elif stored_mtime is None or stored_mtime != stat.st_mtime:
                    await queue.put(SyncTask(
                        "upsert",
                        current_file_id=new_file_id,
                        data_source_id=self._data_source_id,
                    ))
                    n_upserts += 1

        for inode, (fid, _) in by_inode.items():
            if inode not in seen_inodes:
                await queue.put(SyncTask(
                    "delete",
                    current_file_id=fid,
                    data_source_id=self._data_source_id,
                ))
                n_deletes += 1

        if n_skipped:
            log.info(
                "%d files skipped — unsupported file type. "
                "See documentation for the list of supported file types.",
                n_skipped,
            )
        return f"{n_upserts} file upserts  {n_deletes} file deletes"

    async def _watch(self, queue: asyncio.Queue, manifest: "Manifest", catchup_done: asyncio.Event) -> None:
        buffer: list[SyncTask] = []
        async for changes in awatch(self._root_path):
            deleted_paths  = {path for change_type, path in changes if change_type == Change.deleted}
            added_paths    = {path for change_type, path in changes if change_type == Change.added}
            modified_paths = {path for change_type, path in changes if change_type == Change.modified}

            path_to_fid = {file_id_to_path(fid): fid for fid in self._known_file_ids}

            path_by_inode: dict[int, str] = {}
            for p in added_paths:
                p_obj = pathlib.Path(p)
                if p_obj.exists():
                    path_by_inode[p_obj.stat().st_ino] = p

            inode_by_path: dict[str, int] = {p: inode for inode, p in path_by_inode.items()}

            moves: set[tuple[str, str]] = set()
            for old_path in deleted_paths:
                current_fid = path_to_fid.get(old_path)
                if current_fid is None:
                    continue
                try:
                    inode = file_id_to_inode(current_fid)
                except (ValueError, IndexError):
                    continue
                new_path = path_by_inode.get(inode)
                if new_path is not None:
                    moves.add((old_path, new_path))

            moved_old = {old for old, _ in moves}
            moved_new = {new for _, new in moves}

            for old_path, new_path in moves:
                current_fid = path_to_fid[old_path]
                new_file_id = make_file_id(file_id_to_inode(current_fid), new_path)
                self._known_file_ids.discard(current_fid)
                self._known_file_ids.add(new_file_id)
                buffer.append(SyncTask(
                    "delete",
                    current_file_id=current_fid,
                    data_source_id=self._data_source_id,
                ))
                buffer.append(SyncTask(
                    "upsert",
                    current_file_id=new_file_id,
                    data_source_id=self._data_source_id,
                ))

            for path in deleted_paths - moved_old:
                current_fid = path_to_fid.get(path)
                if current_fid is not None:
                    self._known_file_ids.discard(current_fid)
                    buffer.append(SyncTask(
                        "delete",
                        current_file_id=current_fid,
                        data_source_id=self._data_source_id,
                    ))

            for path in (added_paths - moved_new) | modified_paths:
                p = pathlib.Path(path)
                if p.is_file() and self._is_supported(path):
                    inode = inode_by_path.get(path)
                    if inode is None:
                        inode = p.stat().st_ino
                    new_file_id = make_file_id(inode, path)
                    old_fid = path_to_fid.get(path)
                    if old_fid:
                        self._known_file_ids.discard(old_fid)
                    self._known_file_ids.add(new_file_id)
                    buffer.append(SyncTask(
                        "upsert",
                        current_file_id=new_file_id,
                        data_source_id=self._data_source_id,
                    ))

            if catchup_done.is_set():
                for task in buffer:
                    await queue.put(task)
                    log.info(
                        "Change detected: %s",
                        pathlib.Path(file_id_to_path(task.current_file_id)).name,
                    )
                buffer.clear()
