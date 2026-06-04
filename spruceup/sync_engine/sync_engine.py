import logging
from typing import final

from ..connectors.base import TargetConnector
from ..manifest import Manifest
from ..models import ChunkWrapper, SpruceFile

log = logging.getLogger(__name__)


@final
class SyncEngine:
    def __init__(self, manifest: Manifest, target: TargetConnector) -> None:
        self._manifest = manifest
        self._target = target

    async def delete_stale_sources(self, active_ids: list[int]) -> None:
        stale_file_ids = self._manifest.get_orphaned_file_ids(active_ids)
        stale_hashes: set[bytes] = set()
        for file_id in stale_file_ids:
            chunks = self._manifest.get_chunks_for_file(file_id)
            stale_hashes.update(c["content_hash"] for c in chunks)
        target_deletes = [
            h for h in stale_hashes
            if not self._manifest.chunk_hash_referenced_elsewhere(h, stale_file_ids)
        ]

        await self._target.sync([], target_deletes)
        self._manifest.purge_inactive_sources(active_ids)

        log.info("Deleted %d stale chunk(s) from target db", len(target_deletes))

    async def reconcile(self, file: SpruceFile) -> None:
        manifest_upserts: list[tuple[str, ChunkWrapper]] = []
        target_upserts: list[ChunkWrapper] = []
        manifest_deletes: list[tuple[str, bytes]] = []
        target_deletes: list[bytes] = []

        prev_chunks = self._manifest.get_chunks_for_file(file.file_id)
        prev_hashes: set[bytes] = {c["content_hash"] for c in prev_chunks}
        curr_hashes: dict[bytes, ChunkWrapper] = {
            chunk.user_chunk_object_hash: chunk for chunk in file.chunks
        }

        for h, chunk in curr_hashes.items():
            if file.force_upsert or h not in prev_hashes:
                manifest_upserts.append((file.file_id, chunk))
                target_upserts.append(chunk)

        for h in prev_hashes:
            if h not in curr_hashes:
                manifest_deletes.append((file.file_id, h))
                if not self._manifest.chunk_hash_referenced_elsewhere(h, [file.file_id]):
                    target_deletes.append(h)

        self._manifest.ensure_file_row_exists(file.file_id)

        await self._target.sync(target_upserts, target_deletes)

        with self._manifest.transaction():
            self._manifest.upsert_chunks(manifest_upserts)
            self._manifest.delete_chunks(manifest_deletes)
            self._manifest.upsert_file_row(file)

        log.info(
            "Synced %s — %d upserted  %d deleted",
            file.display_name,
            len(target_upserts),
            len(target_deletes),
        )

    async def move_file(self, old_file_id: str, new_file_id: str) -> None:
        self._manifest.update_file_id(old_file_id, new_file_id)
        log.info("Moved manifest row: %s → %s", old_file_id, new_file_id)

    async def delete_file(self, file_id: str) -> None:
        chunks = self._manifest.get_chunks_for_file(file_id)
        content_hashes = [
            c["content_hash"] for c in chunks
            if not self._manifest.chunk_hash_referenced_elsewhere(c["content_hash"], [file_id])
        ]

        await self._target.sync([], content_hashes)
        self._manifest.delete_file_row(file_id)
        log.info("Deleted %d chunk(s) for file_id=%s", len(content_hashes), file_id)
