import sqlite3
from typing import Type

import psycopg

from . import manifest as manifest_db
from . import target_db
from .models import ChunkWrapper, File, TargetTableConfig, UserDefinedChunkSchema


class SyncEngine:
    def __init__(self, manifest_path: str, pg_connstr: str) -> None:
        self._manifest_path = manifest_path
        self._pg_connstr = pg_connstr
        self._config: TargetTableConfig | None = None
        with sqlite3.connect(manifest_path) as conn:
            manifest_db.init_schema(conn)

    def define_target_table(
        self,
        db_name: str,
        table_name: str,
        schema_from_class: Type[UserDefinedChunkSchema],
        primary_key: str,
    ) -> None:
        self._config = TargetTableConfig(
            db_name=db_name,
            table_name=table_name,
            schema_class=schema_from_class,
            primary_key=primary_key,
        )

    def reconcile(self, files: list[File]) -> None:
        """Upsert new/changed chunks, delete orphaned chunks, then stamp each file row."""
        assert self._config is not None, "Call define_target_table() before reconcile()"

        # (file_id, chunk) pairs so manifest.upsert_chunks knows which file each chunk belongs to
        target_upserts: list[tuple[bytes, ChunkWrapper]] = []
        manifest_deletes: list[bytes] = []  # our internal chunk_ids
        pg_deletes: list = []               # user PKs for the Postgres DELETE

        with sqlite3.connect(self._manifest_path) as manifest_conn:
            for file in files:
                prev_chunks = manifest_db.get_chunks_for_file(
                    manifest_conn, file.file_id, self._config.primary_key
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
                        target_upserts.append((file.file_id, curr))
                    elif curr is None:
                        manifest_deletes.append(prev["manifest_chunk_id"])
                        pg_deletes.append(prev["user_pk"])
                    elif prev["user_chunk_object_hash"] != curr.user_chunk_object_hash:
                        target_upserts.append((file.file_id, curr))

            for file in files:
                manifest_db.ensure_file_row_exists(manifest_conn, file.file_id)

            with psycopg.connect(self._pg_connstr) as pg_conn:
                target_db.ensure_table_exists(pg_conn, self._config)
                target_db.upsert_chunks(pg_conn, [c for _, c in target_upserts], self._config)
                target_db.delete_chunks(pg_conn, pg_deletes, self._config)

            manifest_db.upsert_chunks(manifest_conn, target_upserts)
            manifest_db.delete_chunks(manifest_conn, manifest_deletes)

            for file in files:
                manifest_db.upsert_file_row(manifest_conn, file)

    def delete_file(self, file_id: bytes) -> None:
        """Delete all vectors for a file that has been removed from the corpus."""
        assert self._config is not None, "Call define_target_table() before delete_file()"

        with sqlite3.connect(self._manifest_path) as manifest_conn:
            chunks = manifest_db.get_chunks_for_file(
                manifest_conn, file_id, self._config.primary_key
            )
            manifest_chunk_ids = [c["manifest_chunk_id"] for c in chunks]
            pg_pks = [c["user_pk"] for c in chunks]

            with psycopg.connect(self._pg_connstr) as pg_conn:
                target_db.delete_chunks(pg_conn, pg_pks, self._config)

            manifest_db.delete_chunks(manifest_conn, manifest_chunk_ids)
            manifest_db.delete_file_row(manifest_conn, file_id)
