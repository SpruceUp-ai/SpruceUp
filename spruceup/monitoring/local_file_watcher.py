import asyncio
import logging
import pathlib
import time

from watchfiles import awatch, Change

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
        self._known_file_ids: set[str] = set()

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
        n_upserts = n_moves = n_deletes = 0
        use_manifest_cache = (
            force_reindex and manifest.get_config_value("file_cache_ready") == "true"
        )

        by_inode: dict[int, tuple[str, float | None]] = {}
        if not force_reindex:
            for rec in manifest.get_files_for_source(self._data_source_id):
                fid = rec["file_id"]
                try:
                    by_inode[int(fid.split(":", 1)[0])] = (fid, rec["modified_at"])
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
            new_file_id = f"{inode}:{path_str}"
            seen_inodes.add(inode)
            self._known_file_ids.add(new_file_id)

            if force_reindex:
                await queue.put(SyncTask(
                    self._source_type, "upsert", stat.st_mtime,
                    current_file_id=new_file_id,
                    data_source_id=self._data_source_id,
                    use_manifest_cache=use_manifest_cache,
                ))
                n_upserts += 1
                continue

            stored = by_inode.get(inode)
            if stored is None:
                await queue.put(SyncTask(
                    self._source_type, "upsert", stat.st_mtime,
                    current_file_id=new_file_id,
                    data_source_id=self._data_source_id,
                ))
                n_upserts += 1
            else:
                stored_file_id, stored_mtime = stored
                renamed = stored_file_id != new_file_id
                if stored_mtime is not None and stored_mtime == stat.st_mtime:
                    if renamed:
                        await queue.put(SyncTask(
                            self._source_type, "move", stat.st_mtime,
                            current_file_id=stored_file_id,
                            new_file_id=new_file_id,
                            data_source_id=self._data_source_id,
                        ))
                        n_moves += 1
                else:
                    if renamed:
                        await queue.put(SyncTask(
                            self._source_type, "move", stat.st_mtime,
                            current_file_id=stored_file_id,
                            new_file_id=new_file_id,
                            data_source_id=self._data_source_id,
                        ))
                        n_moves += 1
                    await queue.put(SyncTask(
                        self._source_type, "upsert", stat.st_mtime,
                        current_file_id=new_file_id,
                        data_source_id=self._data_source_id,
                    ))
                    n_upserts += 1

        for inode, (fid, stored_mtime) in by_inode.items():
            if inode not in seen_inodes:
                await queue.put(SyncTask(
                    self._source_type, "delete",
                    stored_mtime if stored_mtime is not None else time.time(),
                    current_file_id=fid,
                    data_source_id=self._data_source_id,
                ))
                n_deletes += 1

        log.info(
            "Catch-up complete — %d upsert(s)  %d move(s)  %d delete(s)  %d skipped",
            n_upserts, n_moves, n_deletes, n_skipped,
        )
        if n_skipped:
            log.info(
                "%d file(s) were skipped due to unsupported file type. "
                "See documentation for the list of supported file types.",
                n_skipped,
            )

    async def _watch(self, queue: asyncio.Queue, manifest: "Manifest", catchup_done: asyncio.Event) -> None:
        buffer: list[SyncTask] = []
        async for changes in awatch(self._root_path):
            deleted_paths  = {path for change_type, path in changes if change_type == Change.deleted}
            added_paths    = {path for change_type, path in changes if change_type == Change.added}
            modified_paths = {path for change_type, path in changes if change_type == Change.modified}

            path_to_fid = {fid.split(":", 1)[1]: fid for fid in self._known_file_ids}

            # Stat each added path once, capturing inode + mtime to avoid double-stat
            added_stats: dict[int, tuple[str, float]] = {}
            for p in added_paths:
                p_obj = pathlib.Path(p)
                if p_obj.exists():
                    st = p_obj.stat()
                    added_stats[st.st_ino] = (p, st.st_mtime)

            # Reverse map for the upsert loop: O(1) path lookup for added files
            added_path_stat: dict[str, tuple[int, float]] = {
                p: (inode, mtime) for inode, (p, mtime) in added_stats.items()
            }

            moves: set[tuple[str, str]] = set()
            for old_path in deleted_paths:
                current_fid = path_to_fid.get(old_path)
                if current_fid is None:
                    continue
                try:
                    inode = int(current_fid.split(":", 1)[0])
                except (ValueError, IndexError):
                    continue
                result = added_stats.get(inode)
                if result is not None:
                    moves.add((old_path, result[0]))

            moved_old = {old for old, _ in moves}
            moved_new = {new for _, new in moves}

            for old_path, new_path in moves:
                current_fid = path_to_fid[old_path]
                inode_str = current_fid.split(":", 1)[0]
                _, mtime = added_stats[int(inode_str)]
                new_file_id = f"{inode_str}:{new_path}"
                self._known_file_ids.discard(current_fid)
                self._known_file_ids.add(new_file_id)
                buffer.append(SyncTask(
                    self._source_type, "move", mtime,
                    current_file_id=current_fid,
                    new_file_id=new_file_id,
                    data_source_id=self._data_source_id,
                ))

            for path in deleted_paths - moved_old:
                current_fid = path_to_fid.get(path)
                if current_fid is not None:
                    self._known_file_ids.discard(current_fid)
                    buffer.append(SyncTask(
                        self._source_type, "delete", time.time(),
                        current_file_id=current_fid,
                        data_source_id=self._data_source_id,
                    ))

            for path in (added_paths - moved_new) | modified_paths:
                p = pathlib.Path(path)
                if p.is_file() and self._is_supported(path):
                    cached = added_path_stat.get(path)
                    if cached is not None:
                        inode, mtime = cached
                    else:
                        st = p.stat()
                        inode, mtime = st.st_ino, st.st_mtime
                    new_file_id = f"{inode}:{path}"
                    old_fid = path_to_fid.get(path)
                    if old_fid:
                        self._known_file_ids.discard(old_fid)
                    self._known_file_ids.add(new_file_id)
                    buffer.append(SyncTask(
                        self._source_type, "upsert", mtime,
                        current_file_id=new_file_id,
                        data_source_id=self._data_source_id,
                    ))

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
