import logging
from typing import final

from ..connectors.base import TargetConnector
from ..manifest import Manifest
from ..models import ChunkWrapper, SpruceFile
from ..utils.hashing import hash_source_ref

log = logging.getLogger(__name__)


@final
class SyncEngine:
    def __init__(self, manifest: Manifest, target: TargetConnector) -> None:
        self._manifest = manifest
        self._target = target

    async def delete_stale_sources(self, active_ids: list[int]) -> None:
        target_deletes: list[bytes] = []
        with self._manifest.connect() as conn:
            stale_file_ids = self._manifest.get_orphaned_file_ids(conn, active_ids)
            for file_id in stale_file_ids:
                chunks = self._manifest.get_chunks_for_file(conn, file_id)
                target_deletes.extend(c["content_hash"] for c in chunks)

        await self._target.sync([], target_deletes)

        with self._manifest.connect() as conn:
            self._manifest.purge_inactive_sources(conn, active_ids)

        log.info("Deleted %d stale chunk(s) from target db", len(target_deletes))

    async def reconcile(self, files: list[SpruceFile]) -> None:
        manifest_upserts: list[tuple[bytes, ChunkWrapper]] = []
        target_upserts: list[ChunkWrapper] = []
        manifest_deletes: list[tuple[bytes, bytes]] = []  # (file_id, content_hash)
        target_deletes: list[bytes] = []  # content hashes

        with self._manifest.connect() as conn:
            for file in files:
                prev_chunks = self._manifest.get_chunks_for_file(conn, file.file_id)
                prev_hashes: set[bytes] = {c["content_hash"] for c in prev_chunks}
                curr_hashes: dict[bytes, ChunkWrapper] = {
                    chunk.user_chunk_object_hash: chunk for chunk in file.chunks
                }

                for h, chunk in curr_hashes.items():
                    if h not in prev_hashes:
                        manifest_upserts.append((file.file_id, chunk))
                        target_upserts.append(chunk)

                for h in prev_hashes:
                    if h not in curr_hashes:
                        manifest_deletes.append((file.file_id, h))
                        target_deletes.append(h)

            for file in files:
                self._manifest.ensure_file_row_exists(
                    conn, file.file_id, file.source_ref
                )

        await self._target.sync(target_upserts, target_deletes)

        with self._manifest.connect() as conn:
            self._manifest.upsert_chunks(conn, manifest_upserts)
            self._manifest.delete_chunks(conn, manifest_deletes)

            for file in files:
                self._manifest.upsert_file_row(conn, file)
                self._manifest.upsert_file_metadata(
                    conn, file.file_id, file.source_metadata
                )

        log.info(
            "Synced %s — %d upserted  %d deleted",
            ", ".join(f.source_ref for f in files),
            len(target_upserts),
            len(target_deletes),
        )

    async def move_file(self, old_ref: str, new_ref: str) -> None:
        old_file_id = hash_source_ref(old_ref)
        new_file_id = hash_source_ref(new_ref)
        with self._manifest.connect() as conn:
            self._manifest.move_file_row(conn, old_file_id, new_file_id, new_ref)
        log.info("Moved manifest row: %s → %s", old_ref, new_ref)

    async def delete_file(self, source_ref: str) -> None:
        file_id = hash_source_ref(source_ref)
        with self._manifest.connect() as conn:
            chunks = self._manifest.get_chunks_for_file(conn, file_id)
        content_hashes = [c["content_hash"] for c in chunks]

        await self._target.sync([], content_hashes)

        with self._manifest.connect() as conn:
            self._manifest.delete_file_row(conn, file_id)
        log.info("Deleted %d chunk(s) for %s", len(content_hashes), source_ref)
