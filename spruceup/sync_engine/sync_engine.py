import logging
import pathlib

from ..connectors.base import TargetConnector
from ..manifest import Manifest
from ..models import ChunkWrapper, SpruceFile
from ..utils.hashing import hash_file_path

log = logging.getLogger(__name__)


class SyncEngine:
    def __init__(self, manifest: Manifest, target: TargetConnector) -> None:
        self._manifest = manifest
        self._target = target

    def delete_stale_sources(self, active_ids: list[int]) -> None:
        target_deletes: list = []
        with self._manifest.connect() as conn:
            stale_file_ids = self._manifest.get_stale_file_ids(conn, active_ids)
            for file_id in stale_file_ids:
                chunks = self._manifest.get_chunks_for_file(
                    conn, file_id, self._target.primary_key
                )
                target_deletes.extend(chunk["user_pk"] for chunk in chunks)

            self._target.sync([], target_deletes)
            self._manifest.delete_stale_data_sources(conn, active_ids)

        log.info("Deleted %d stale chunk(s) from target db", len(target_deletes))

    def reconcile(self, files: list[SpruceFile]) -> None:
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
                    conn, file.file_id, file.file_path
                )

            self._target.sync(target_upserts, target_deletes)

            self._manifest.upsert_chunks(conn, manifest_upserts)
            self._manifest.delete_chunks(conn, manifest_deletes)

            for file in files:
                self._manifest.upsert_file_row(conn, file)

        log.info(
            "Synced %s — %d upserted  %d deleted",
            ", ".join(pathlib.Path(f.file_path).name for f in files),
            len(target_upserts),
            len(target_deletes),
        )

    async def move_file(self, old_path: str, new_path: str) -> None:
        old_file_id = hash_file_path(old_path)
        new_file_id = hash_file_path(new_path)
        with self._manifest.connect() as conn:
            self._manifest.move_file_row(conn, old_file_id, new_file_id, new_path)
        log.info(
            "Moved manifest row: %s → %s",
            pathlib.Path(old_path).name,
            pathlib.Path(new_path).name,
        )

    async def delete_file(self, file_path: str) -> None:
        file_id = hash_file_path(file_path)
        with self._manifest.connect() as conn:
            chunks = self._manifest.get_chunks_for_file(
                conn, file_id, self._target.primary_key
            )
            manifest_chunk_ids = [c["manifest_chunk_id"] for c in chunks]
            target_pks = [c["user_pk"] for c in chunks]

            self._target.sync([], target_pks)

            self._manifest.delete_chunks(conn, manifest_chunk_ids)
            self._manifest.delete_file_row(conn, file_id)
        log.info(
            "Deleted %d chunk(s) for %s", len(target_pks), pathlib.Path(file_path).name
        )
