import logging

from ..connectors.base import TargetConnector
from ..manifest import Manifest
from ..models import ChunkWrapper, SpruceFile
from ..utils.hashing import hash_source_ref

log = logging.getLogger(__name__)


class SyncEngine:
    def __init__(self, manifest: Manifest, target: TargetConnector) -> None:
        self._manifest = manifest
        self._target = target

    async def delete_stale_sources(self, active_ids: list[int]) -> None:
        target_deletes: list = []
        with self._manifest.connect() as conn:
            stale_file_ids = self._manifest.get_stale_file_ids(conn, active_ids)
            for file_id in stale_file_ids:
                chunks = self._manifest.get_chunks_for_file(
                    conn, file_id, self._target.primary_key
                )
                target_deletes.extend(chunk["user_pk"] for chunk in chunks)

        await self._target.sync([], target_deletes)

        with self._manifest.connect() as conn:
            self._manifest.delete_stale_data_sources(conn, active_ids)

        log.info("Deleted %d stale chunk(s) from target db", len(target_deletes))

    async def reconcile(self, files: list[SpruceFile]) -> None:
        manifest_upserts: list[tuple[bytes, ChunkWrapper]] = []
        target_upserts: list[ChunkWrapper] = []
        manifest_deletes: list[bytes] = []
        target_deletes: list = []

        with self._manifest.connect() as conn:
            for file in files:
                prev_chunks = self._manifest.get_chunks_for_file(
                    conn, file.file_id, self._target.primary_key
                )

                chunks_by_id: dict[bytes, dict] = {
                    p["manifest_chunk_id"]: {"prev": p, "curr": None}
                    for p in prev_chunks
                }

                for chunk in file.chunks:
                    key = chunk.chunk_id
                    if key in chunks_by_id:
                        chunks_by_id[key]["curr"] = chunk
                    else:
                        chunks_by_id[key] = {"prev": None, "curr": chunk}

                for pair in chunks_by_id.values():
                    prev, curr = pair["prev"], pair["curr"]
                    if prev is None:
                        manifest_upserts.append((file.file_id, curr))
                        target_upserts.append(curr)
                    elif curr is None:
                        manifest_deletes.append(prev["manifest_chunk_id"])
                        target_deletes.append(prev["user_pk"])
                    elif prev["user_chunk_object_hash"] != curr.user_chunk_object_hash:
                        manifest_upserts.append((file.file_id, curr))
                        target_upserts.append(curr)

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
                self._manifest.upsert_file_metadata(conn, file.file_id, file.source_metadata)

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
            chunks = self._manifest.get_chunks_for_file(
                conn, file_id, self._target.primary_key
            )
        manifest_chunk_ids = [c["manifest_chunk_id"] for c in chunks]
        target_pks = [c["user_pk"] for c in chunks]

        await self._target.sync([], target_pks)

        with self._manifest.connect() as conn:
            self._manifest.delete_chunks(conn, manifest_chunk_ids)
            self._manifest.delete_file_row(conn, file_id)
        log.info("Deleted %d chunk(s) for %s", len(target_pks), source_ref)
