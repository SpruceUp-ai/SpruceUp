import dataclasses
import json
import sqlite3

from .models import ChunkWrapper, File


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS data_sources (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type VARCHAR(25) NOT NULL
        );

        CREATE TABLE IF NOT EXISTS files (
            id             BLOB PRIMARY KEY,
            transform_hash BLOB,
            content_hash   BLOB,
            mtime          REAL,
            data_source_id INTEGER REFERENCES data_sources(id) ON DELETE CASCADE,
            file_type      VARCHAR(10)
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id                      BLOB PRIMARY KEY,
            file_id                 BLOB REFERENCES files(id) ON DELETE CASCADE,
            transform_hash          BLOB,
            user_chunk_object_hash  BLOB,
            user_chunk_object       BLOB
        );
    """)


def get_chunks_for_file(conn: sqlite3.Connection, file_id: bytes, pk_col: str) -> list[dict]:
    """Return all manifest chunk records for a file.

    Each record contains:
      manifest_chunk_id      – our internal chunk_id (bytes), the manifest PK
      user_chunk_object_hash – for change-detection in reconcile
      user_pk                – the user's primary key value, needed for Postgres deletes
    """
    cursor = conn.execute(
        "SELECT id, user_chunk_object_hash, user_chunk_object FROM chunks WHERE file_id = ?",
        (file_id,),
    )
    results = []
    for manifest_chunk_id, obj_hash, obj_blob in cursor:
        user_chunk_data = json.loads(obj_blob.decode())
        results.append({
            "manifest_chunk_id": manifest_chunk_id,
            "user_chunk_object_hash": obj_hash,
            "user_pk": user_chunk_data[pk_col],
        })
    return results


def upsert_chunks(conn: sqlite3.Connection, chunks: list[tuple[bytes, ChunkWrapper]]) -> None:
    if not chunks:
        return
    rows = [
        (
            chunk.chunk_id,
            file_id,
            chunk.user_chunk_object_hash,
            json.dumps(dataclasses.asdict(chunk.user_chunk), default=str).encode(),
        )
        for file_id, chunk in chunks
    ]
    conn.executemany(
        """INSERT OR REPLACE INTO chunks
               (id, file_id, user_chunk_object_hash, user_chunk_object)
           VALUES (?, ?, ?, ?)""",
        rows,
    )


def upsert_file_row(conn: sqlite3.Connection, file: File) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO files
               (id, transform_hash, content_hash, mtime, data_source_id, file_type)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (file.file_id, file.transform_hash, file.content_hash,
         file.mtime, file.data_source_id, file.file_type),
    )


def delete_chunks(conn: sqlite3.Connection, manifest_chunk_ids: list[bytes]) -> None:
    if not manifest_chunk_ids:
        return
    placeholders = ",".join("?" * len(manifest_chunk_ids))
    conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", manifest_chunk_ids)


def delete_file_row(conn: sqlite3.Connection, file_id: bytes) -> None:
    conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
