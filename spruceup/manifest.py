import dataclasses
import json
import sqlite3

from .models import ChunkWrapper, SpruceFile

_MANIFEST_PATH = "spruceup_manifest.db"


class Manifest:
    """Single access point for all SQLite manifest reads and writes."""

    def __init__(self, path: str = _MANIFEST_PATH):
        self._path = path
        self._init_db()

    def _init_db(self) -> None:
        con = self.connect()
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS data_sources (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type       TEXT NOT NULL,
                    source_identifier TEXT NOT NULL,
                    UNIQUE(source_type, source_identifier)
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    id             BLOB PRIMARY KEY,
                    file_path      TEXT NOT NULL,
                    inode          INTEGER,
                    content_hash   BLOB,
                    mtime          REAL,
                    data_source_id INTEGER REFERENCES data_sources(id) ON DELETE CASCADE,
                    file_type      TEXT
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id                     BLOB PRIMARY KEY,
                    file_id                BLOB REFERENCES files(id) ON DELETE CASCADE,
                    user_chunk_object_hash BLOB,
                    user_chunk_object      BLOB
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS transform_hashes (
                    transform_hash BLOB PRIMARY KEY
                )
                """
            )
            con.commit()
        finally:
            con.close()

    def connect(self) -> sqlite3.Connection:
        """Return a connection suitable for use as a context manager (transaction semantics)."""
        conn = sqlite3.connect(self._path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ------------------------------------------------------------------
    # Chunk and file operations (callers manage the connection/transaction)
    # ------------------------------------------------------------------

    def get_chunks_for_file(self, conn: sqlite3.Connection, file_id: bytes, pk_col: str) -> list[dict]:
        """Return all manifest chunk records for a file."""
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

    def upsert_chunks(self, conn: sqlite3.Connection, chunks: list[tuple[bytes, ChunkWrapper]]) -> None:
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

    def ensure_file_row_exists(self, conn: sqlite3.Connection, file_id: bytes, file_path: str) -> None:
        # Insert a skeleton file row so the FK constraint on chunks.file_id is
        # satisfied before chunk writes. Full fields filled by upsert_file_row.
        conn.execute(
            "INSERT OR IGNORE INTO files (id, file_path) VALUES (?, ?)",
            (file_id, file_path),
        )

    def upsert_file_row(self, conn: sqlite3.Connection, file: SpruceFile) -> None:
        # ON CONFLICT DO UPDATE performs an in-place update rather than DELETE+INSERT,
        # so it does not trigger the ON DELETE CASCADE on chunks.file_id.
        conn.execute(
            """INSERT INTO files
                   (id, file_path, inode, content_hash, mtime, data_source_id, file_type)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (id) DO UPDATE SET
                   file_path      = excluded.file_path,
                   inode          = excluded.inode,
                   content_hash   = excluded.content_hash,
                   mtime          = excluded.mtime,
                   data_source_id = excluded.data_source_id,
                   file_type      = excluded.file_type""",
            (
                file.file_id, file.file_path, file.inode,
                file.content_hash, file.mtime, file.data_source_id, file.file_type,
            ),
        )

    def move_file_row(
        self,
        conn: sqlite3.Connection,
        old_file_id: bytes,
        new_file_id: bytes,
        new_path: str,
    ) -> None:
        row = conn.execute(
            "SELECT inode, content_hash, mtime, data_source_id, file_type FROM files WHERE id = ?",
            (old_file_id,),
        ).fetchone()
        if row is None:
            return
        inode, content_hash, mtime, data_source_id, file_type = row
        conn.execute(
            """INSERT INTO files
                   (id, file_path, inode, content_hash, mtime, data_source_id, file_type)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (new_file_id, new_path, inode, content_hash, mtime, data_source_id, file_type),
        )
        conn.execute(
            "UPDATE chunks SET file_id = ? WHERE file_id = ?",
            (new_file_id, old_file_id),
        )
        conn.execute("DELETE FROM files WHERE id = ?", (old_file_id,))

    def delete_chunks(self, conn: sqlite3.Connection, manifest_chunk_ids: list[bytes]) -> None:
        if not manifest_chunk_ids:
            return
        placeholders = ",".join("?" * len(manifest_chunk_ids))
        conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", manifest_chunk_ids)

    def delete_file_row(self, conn: sqlite3.Connection, file_id: bytes) -> None:
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

    # ------------------------------------------------------------------
    # Transform-hash operations (self-contained — manage their own connections)
    # ------------------------------------------------------------------

    def transform_hash_changed(self, transform_hash: bytes) -> bool:
        """Return True if transform_hash is absent from the manifest."""
        con = self.connect()
        try:
            row = con.execute(
                "SELECT 1 FROM transform_hashes WHERE transform_hash = ?", (transform_hash,)
            ).fetchone()
            return row is None
        finally:
            con.close()

    def register_source(self, source_type: str, source_identifier: str) -> int:
        con = self.connect()
        try:
            con.execute(
                "INSERT OR IGNORE INTO data_sources (source_type, source_identifier) VALUES (?, ?)",
                (source_type, source_identifier),
            )
            row = con.execute(
                "SELECT id FROM data_sources WHERE source_type = ? AND source_identifier = ?",
                (source_type, source_identifier),
            ).fetchone()
            con.commit()
            return row[0]
        finally:
            con.close()

    def delete_stale_sources(self, active_ids: list[int]) -> None:
        placeholders = ",".join("?" * len(active_ids))
        con = self.connect()
        try:
            con.execute(
                f"DELETE FROM data_sources WHERE id NOT IN ({placeholders})",
                active_ids,
            )
            con.commit()
        finally:
            con.close()

    def update_transform_hash(self, transform_hash: bytes) -> None:
        """Replace the stored transform hash with the current one."""
        con = self.connect()
        try:
            con.execute("DELETE FROM transform_hashes")
            con.execute(
                "INSERT INTO transform_hashes (transform_hash) VALUES (?)", (transform_hash,)
            )
            con.commit()
        finally:
            con.close()
