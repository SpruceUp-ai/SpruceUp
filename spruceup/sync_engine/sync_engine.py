import logging
import pathlib

import psycopg

from . import target_db
from ..hashing import hash_file_path
from ..manifest import Manifest
from ..models import ChunkWrapper, SpruceFile, TargetTableConfig, UserDefinedChunkSchema

log = logging.getLogger(__name__)


class SyncEngine:
    def __init__(self, manifest: Manifest, pg_connstr: str) -> None:
        self._manifest = manifest
        self._pg_connstr = pg_connstr
        self._config: TargetTableConfig | None = None

    def define_target_table(
        self,
        db_name: str,
        table_name: str,
        schema_from_class: type,
        primary_key: str,
    ) -> None:
        """Register the user's chunk schema and ensure the target Postgres table exists.

        schema_from_class must be a dataclass with at least:
          - chunk_text: str
          - chunk_embedding: list[float]
          - a primary-key field matching primary_key
        """
        self._config = TargetTableConfig(
            db_name=db_name,
            table_name=table_name,
            schema_class=schema_from_class,
            primary_key=primary_key,
        )
        with psycopg.connect(self._pg_connstr) as pg_conn:
            target_db.ensure_table_exists(pg_conn, self._config)

    def reconcile(self, files: list[SpruceFile]) -> None:
        """Upsert new/changed chunks, delete orphaned chunks, then stamp each file row."""
        assert self._config is not None, "Call define_target_table() before reconcile()"

        manifest_upserts: list[tuple[bytes, ChunkWrapper]] = []
        target_upserts: list[ChunkWrapper] = []
        manifest_deletes: list[bytes] = []
        target_deletes: list = []

        with self._manifest.connect() as conn:
            for file in files:
                prev_chunks = self._manifest.get_chunks_for_file(
                    conn, file.file_id, self._config.primary_key
                )

                chunks_by_id: dict[bytes, dict] = {
                    p["manifest_chunk_id"]: {"prev": p, "curr": None} for p in prev_chunks
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
                self._manifest.ensure_file_row_exists(conn, file.file_id, file.file_path)

            with psycopg.connect(self._pg_connstr) as pg_conn:
                # retry:
                target_db.upsert_chunks(pg_conn, target_upserts, self._config)
                # retry:
                target_db.delete_chunks(pg_conn, target_deletes, self._config)

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

    def move_file(self, old_path: str, new_path: str) -> None:
        """Update the manifest when a file is renamed/moved without re-embedding.

        The vectors in Postgres are keyed by user-defined primary keys (content-based),
        so they remain valid after a rename. Only the SQLite manifest needs updating.
        """
        old_file_id = hash_file_path(old_path)
        new_file_id = hash_file_path(new_path)
        with self._manifest.connect() as conn:
            self._manifest.move_file_row(conn, old_file_id, new_file_id, new_path)
        log.info(
            "Moved manifest row: %s → %s",
            pathlib.Path(old_path).name,
            pathlib.Path(new_path).name,
        )

    def delete_file(self, file_path: str) -> None:
        """Delete all vectors for a file that has been removed from the corpus."""
        assert self._config is not None, "Call define_target_table() before delete_file()"

        file_id = hash_file_path(file_path)
        with self._manifest.connect() as conn:
            chunks = self._manifest.get_chunks_for_file(
                conn, file_id, self._config.primary_key
            )
            manifest_chunk_ids = [c["manifest_chunk_id"] for c in chunks]
            pg_pks = [c["user_pk"] for c in chunks]

            with psycopg.connect(self._pg_connstr) as pg_conn:
                target_db.delete_chunks(pg_conn, pg_pks, self._config)

            self._manifest.delete_chunks(conn, manifest_chunk_ids)
            self._manifest.delete_file_row(conn, file_id)
        log.info("Deleted %d chunk(s) for %s", len(pg_pks), pathlib.Path(file_path).name)
