import dataclasses
import json
import sqlite3

from models import ChunkWrapper, SpruceFile


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


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


def ensure_file_row_exists(conn: sqlite3.Connection, file_id: bytes, file_path: str) -> None:
    """Insert a skeleton file row if one does not already exist.

    Called before writing chunks so the FK constraint on chunks.file_id is
    satisfied immediately. Nullable columns are left NULL and filled in by
    upsert_file_row at the end of reconcile.
    """
    conn.execute(
        "INSERT OR IGNORE INTO files (id, file_path) VALUES (?, ?)",
        (file_id, file_path),
    )


def upsert_file_row(conn: sqlite3.Connection, file: SpruceFile) -> None:
    # ON CONFLICT DO UPDATE performs an in-place update rather than DELETE+INSERT,
    # so it does not trigger the ON DELETE CASCADE on chunks.file_id.
    conn.execute(
        """INSERT INTO files
               (id, file_path, inode, transform_hash, content_hash, mtime, data_source_id, file_type)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (id) DO UPDATE SET
               file_path      = excluded.file_path,
               inode          = excluded.inode,
               transform_hash = excluded.transform_hash,
               content_hash   = excluded.content_hash,
               mtime          = excluded.mtime,
               data_source_id = excluded.data_source_id,
               file_type      = excluded.file_type""",
        (
            file.file_id, file.file_path, file.inode,
            file.transform_hash, file.content_hash,
            file.mtime, file.data_source_id, file.file_type,
        ),
    )


def move_file_row(conn: sqlite3.Connection, old_file_id: bytes, new_file_id: bytes, new_path: str) -> None:
    """Rename a file in the manifest without touching Postgres or re-embedding.

    Steps:
      1. Copy the old file row under the new id/path.
      2. Re-point all chunk rows to the new file id.
      3. Delete the old file row (chunks already moved, so no cascade).
    """
    row = conn.execute(
        "SELECT inode, transform_hash, content_hash, mtime, data_source_id, file_type FROM files WHERE id = ?",
        (old_file_id,),
    ).fetchone()
    if row is None:
        return
    inode, transform_hash, content_hash, mtime, data_source_id, file_type = row
    conn.execute(
        """INSERT OR REPLACE INTO files
               (id, file_path, inode, transform_hash, content_hash, mtime, data_source_id, file_type)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (new_file_id, new_path, inode, transform_hash, content_hash, mtime, data_source_id, file_type),
    )
    conn.execute(
        "UPDATE chunks SET file_id = ? WHERE file_id = ?",
        (new_file_id, old_file_id),
    )
    conn.execute("DELETE FROM files WHERE id = ?", (old_file_id,))


def delete_chunks(conn: sqlite3.Connection, manifest_chunk_ids: list[bytes]) -> None:
    if not manifest_chunk_ids:
        return
    placeholders = ",".join("?" * len(manifest_chunk_ids))
    conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", manifest_chunk_ids)


def delete_file_row(conn: sqlite3.Connection, file_id: bytes) -> None:
    conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
