import asyncio
import logging
import pathlib

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
        # Set of file_ids currently tracked; each encodes inode + path as f"{inode}:{path}"
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

        # inode → (file_id, modified_at) for O(1) per-file lookup during scan
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
                    self._source_type, path_str, "upsert",
                    current_file_id=new_file_id,
                    data_source_id=self._data_source_id,
                    use_manifest_cache=use_manifest_cache,
                ))
                n_upserts += 1
                continue

            stored = by_inode.get(inode)
            if stored is None:
                await queue.put(SyncTask(
                    self._source_type, path_str, "upsert",
                    current_file_id=new_file_id,
                    data_source_id=self._data_source_id,
                ))
                n_upserts += 1
            else:
                stored_file_id, stored_mtime = stored
                renamed = stored_file_id != new_file_id
                if stored_mtime is not None and stored_mtime == stat.st_mtime:
                    if renamed:
                        # Same inode, same mtime, path changed → rename only, no re-embed
                        await queue.put(SyncTask(
                            self._source_type, path_str, "move",
                            current_file_id=stored_file_id,
                            new_file_id=new_file_id,
                            data_source_id=self._data_source_id,
                        ))
                        n_moves += 1
                else:
                    # mtime changed or unknown → content may have changed
                    if renamed:
                        await queue.put(SyncTask(
                            self._source_type, path_str, "move",
                            current_file_id=stored_file_id,
                            new_file_id=new_file_id,
                            data_source_id=self._data_source_id,
                        ))
                        n_moves += 1
                    await queue.put(SyncTask(
                        self._source_type, path_str, "upsert",
                        current_file_id=new_file_id,
                        data_source_id=self._data_source_id,
                    ))
                    n_upserts += 1

        for inode, (fid, _) in by_inode.items():
            if inode not in seen_inodes:
                old_path = fid.split(":", 1)[1] if ":" in fid else fid
                await queue.put(SyncTask(
                    self._source_type, old_path, "delete",
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

            # Build path→file_id index from the set for this batch
            path_to_fid = {fid.split(":", 1)[1]: fid for fid in self._known_file_ids}

            added_by_inode: dict[int, str] = {
                pathlib.Path(p).stat().st_ino: p
                for p in added_paths
                if pathlib.Path(p).exists()
            }

            # Detect renames: a deleted path whose inode reappears under a new path
            moves: set[tuple[str, str]] = set()  # {(old_path, new_path)}
            for old_path in deleted_paths:
                current_fid = path_to_fid.get(old_path)
                if current_fid is None:
                    continue
                try:
                    inode = int(current_fid.split(":", 1)[0])
                except (ValueError, IndexError):
                    continue
                new_path = added_by_inode.get(inode)
                if new_path is not None:
                    moves.add((old_path, new_path))

            moved_old = {old for old, _ in moves}
            moved_new = {new for _, new in moves}

            for old_path, new_path in moves:
                current_fid = path_to_fid[old_path]
                new_file_id = f"{current_fid.split(':', 1)[0]}:{new_path}"
                self._known_file_ids.discard(current_fid)
                self._known_file_ids.add(new_file_id)
                buffer.append(SyncTask(
                    self._source_type, new_path, "move",
                    current_file_id=current_fid,
                    new_file_id=new_file_id,
                    data_source_id=self._data_source_id,
                ))

            for path in deleted_paths - moved_old:
                current_fid = path_to_fid.get(path)
                if current_fid is not None:
                    self._known_file_ids.discard(current_fid)
                    buffer.append(SyncTask(
                        self._source_type, path, "delete",
                        current_file_id=current_fid,
                        data_source_id=self._data_source_id,
                    ))

            for path in (added_paths - moved_new) | modified_paths:
                p = pathlib.Path(path)
                if p.is_file() and self._is_supported(path):
                    stat = p.stat()
                    new_file_id = f"{stat.st_ino}:{path}"
                    old_fid = path_to_fid.get(path)
                    if old_fid:
                        self._known_file_ids.discard(old_fid)
                    self._known_file_ids.add(new_file_id)
                    buffer.append(SyncTask(
                        self._source_type, path, "upsert",
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
