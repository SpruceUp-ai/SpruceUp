import dataclasses
import json
import sqlite3

from .models import ChunkWrapper, File


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


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


def ensure_file_row_exists(conn: sqlite3.Connection, file_id: bytes) -> None:
    """Insert a skeleton file row if one does not already exist.

    Called before writing chunks so the FK constraint on chunks.file_id is
    satisfied immediately. The row's other columns are left NULL and filled in
    by upsert_file_row at the end of reconcile.
    """
    conn.execute("INSERT OR IGNORE INTO files (id) VALUES (?)", (file_id,))


def upsert_file_row(conn: sqlite3.Connection, file: File) -> None:
    # ON CONFLICT DO UPDATE performs an in-place update rather than DELETE+INSERT,
    # so it does not trigger the ON DELETE CASCADE on chunks.file_id.
    conn.execute(
        """INSERT INTO files (id, transform_hash, content_hash, mtime, data_source_id, file_type)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT (id) DO UPDATE SET
                   transform_hash = excluded.transform_hash,
                   content_hash   = excluded.content_hash,
                   mtime          = excluded.mtime,
                   data_source_id = excluded.data_source_id,
                   file_type      = excluded.file_type""",
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
