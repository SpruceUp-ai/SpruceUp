import dataclasses
import json
import sqlite3

from .models import ChunkWrapper, SpruceFile

_MANIFEST_PATH = "spruceup_manifest.db"


class Manifest:
    """Single access point for all SQLite manifest reads and writes."""

    def __init__(self, path: str = _MANIFEST_PATH):
        self._path = path
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_db()

    # ------------------------------------------------------------------
    # Schema init
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._conn:
            self._init_sources_schema(self._conn)
            self._init_files_schema(self._conn)
            self._init_chunks_schema(self._conn)
            self._init_transform_schema(self._conn)
            self._init_memoize_schema(self._conn)

    def _init_sources_schema(self, con: sqlite3.Connection) -> None:
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
            CREATE TABLE IF NOT EXISTS source_state (
                data_source_id INTEGER NOT NULL
                    REFERENCES data_sources(id) ON DELETE CASCADE,
                key            TEXT NOT NULL,
                value          TEXT NOT NULL,
                PRIMARY KEY (data_source_id, key)
            )
            """
        )

    def _init_files_schema(self, con: sqlite3.Connection) -> None:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                id             BLOB PRIMARY KEY,
                source_ref     TEXT NOT NULL,
                content_hash   BLOB,
                data_source_id INTEGER REFERENCES data_sources(id) ON DELETE CASCADE,
                file_type      TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS file_metadata (
                file_id BLOB NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                key     TEXT NOT NULL,
                value   TEXT NOT NULL,
                PRIMARY KEY (file_id, key)
            )
            """
        )

    def _init_chunks_schema(self, con: sqlite3.Connection) -> None:
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
        # reconcile/delete look up chunks by file_id on every file; without this
        # index that is a full scan of the growing chunks table (O(N^2) ingest).
        con.execute(
            "CREATE INDEX IF NOT EXISTS ix_chunks_file_id ON chunks(file_id)"
        )

    def _init_transform_schema(self, con: sqlite3.Connection) -> None:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS transform_hashes (
                transform_hash BLOB PRIMARY KEY
            )
            """
        )

    def _init_memoize_schema(self, con: sqlite3.Connection) -> None:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS memoize_cache (
                file_id   BLOB NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                fn_hash   BLOB NOT NULL,
                args_hash BLOB NOT NULL,
                result    BLOB NOT NULL,
                PRIMARY KEY (file_id, fn_hash, args_hash)
            )
            """
        )

    def connect(self) -> sqlite3.Connection:
        """Return the process-wide shared SQLite connection.

        The Manifest keeps a single connection for the whole pipeline process —
        opening a fresh one per call was, at three connections per file, weighing
        heavily on RAM as the corpus grew. Use it directly, or as a context
        manager (`with manifest.connect() as conn:`) to group writes into one
        transaction. Do not close it; the Manifest owns its lifetime.

        Safe to share because every manifest DB section is synchronous — no two
        coroutines interleave a transaction on the connection.
        """
        return self._conn

    # ------------------------------------------------------------------
    # Chunk and file operations (callers manage the connection/transaction)
    # ------------------------------------------------------------------

    def get_chunks_for_file(
        self, conn: sqlite3.Connection, file_id: bytes, pk_col: str
    ) -> list[dict]:
        """Return all manifest chunk records for a file."""
        cursor = conn.execute(
            "SELECT id, user_chunk_object_hash, user_chunk_object FROM chunks WHERE file_id = ?",
            (file_id,),
        )
        results = []
        for manifest_chunk_id, obj_hash, obj_blob in cursor:
            user_chunk_data = json.loads(obj_blob.decode())
            results.append(
                {
                    "manifest_chunk_id": manifest_chunk_id,
                    "user_chunk_object_hash": obj_hash,
                    "user_pk": user_chunk_data[pk_col],
                }
            )
        return results

    def upsert_chunks(
        self, conn: sqlite3.Connection, chunks: list[tuple[bytes, ChunkWrapper]]
    ) -> None:
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

    def ensure_file_row_exists(
        self, conn: sqlite3.Connection, file_id: bytes, source_ref: str
    ) -> None:
        # Insert a skeleton file row so the FK constraint on chunks.file_id is
        # satisfied before chunk writes. Full fields filled by upsert_file_row.
        conn.execute(
            "INSERT OR IGNORE INTO files (id, source_ref) VALUES (?, ?)",
            (file_id, source_ref),
        )

    def upsert_file_row(self, conn: sqlite3.Connection, file: SpruceFile) -> None:
        # ON CONFLICT DO UPDATE performs an in-place update rather than DELETE+INSERT,
        # so it does not trigger the ON DELETE CASCADE on chunks.file_id.
        conn.execute(
            """INSERT INTO files
                   (id, source_ref, content_hash, data_source_id, file_type)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT (id) DO UPDATE SET
                   source_ref     = excluded.source_ref,
                   content_hash   = excluded.content_hash,
                   data_source_id = excluded.data_source_id,
                   file_type      = excluded.file_type""",
            (
                file.file_id,
                file.source_ref,
                file.content_hash,
                file.data_source_id,
                file.file_type,
            ),
        )

    def upsert_file_metadata(
        self, conn: sqlite3.Connection, file_id: bytes, metadata: dict
    ) -> None:
        if not metadata:
            return
        conn.executemany(
            "INSERT OR REPLACE INTO file_metadata (file_id, key, value) VALUES (?, ?, ?)",
            [(file_id, k, str(v)) for k, v in metadata.items()],
        )

    def get_file_metadata(self, conn: sqlite3.Connection, file_id: bytes) -> dict:
        rows = conn.execute(
            "SELECT key, value FROM file_metadata WHERE file_id = ?",
            (file_id,),
        ).fetchall()
        return {key: value for key, value in rows}

    def get_source_refs(self, conn: sqlite3.Connection, data_source_id: int) -> set[str]:
        """Return the set of source_refs currently tracked for a source."""
        rows = conn.execute(
            "SELECT source_ref FROM files WHERE data_source_id = ?",
            (data_source_id,),
        ).fetchall()
        return {row[0] for row in rows}

    def get_files_with_metadata(
        self, conn: sqlite3.Connection, data_source_id: int
    ) -> list[dict]:
        """Return all files for a source with their source_ref, content_hash, and metadata dict."""
        rows = conn.execute(
            "SELECT f.id, f.source_ref, f.content_hash, m.key, m.value "
            "FROM files f "
            "LEFT JOIN file_metadata m ON m.file_id = f.id "
            "WHERE f.data_source_id = ?",
            (data_source_id,),
        ).fetchall()
        files: dict[bytes, dict] = {}
        for file_id, source_ref, content_hash, key, value in rows:
            if file_id not in files:
                files[file_id] = {
                    "source_ref": source_ref,
                    "content_hash": content_hash,
                    "metadata": {},
                }
            if key is not None:
                files[file_id]["metadata"][key] = value
        return list(files.values())

    def move_file_row(
        self,
        conn: sqlite3.Connection,
        old_file_id: bytes,
        new_file_id: bytes,
        new_ref: str,
    ) -> None:
        row = conn.execute(
            "SELECT content_hash, data_source_id, file_type FROM files WHERE id = ?",
            (old_file_id,),
        ).fetchone()
        if row is None:
            return
        content_hash, data_source_id, file_type = row
        conn.execute(
            """INSERT INTO files (id, source_ref, content_hash, data_source_id, file_type)
               VALUES (?, ?, ?, ?, ?)""",
            (new_file_id, new_ref, content_hash, data_source_id, file_type),
        )
        # Copy file_metadata rows to the new file_id
        conn.execute("DELETE FROM file_metadata WHERE file_id = ?", (new_file_id,))
        conn.execute(
            "INSERT OR REPLACE INTO file_metadata "
            "SELECT ?, key, value FROM file_metadata WHERE file_id = ?",
            (new_file_id, old_file_id),
        )
        conn.execute("DELETE FROM memoize_cache WHERE file_id = ?", (new_file_id,))
        conn.execute(
            "UPDATE memoize_cache SET file_id = ? WHERE file_id = ?",
            (new_file_id, old_file_id),
        )
        conn.execute(
            "UPDATE chunks SET file_id = ? WHERE file_id = ?",
            (new_file_id, old_file_id),
        )
        conn.execute("DELETE FROM files WHERE id = ?", (old_file_id,))

    def delete_chunks(
        self, conn: sqlite3.Connection, manifest_chunk_ids: list[bytes]
    ) -> None:
        if not manifest_chunk_ids:
            return
        placeholders = ",".join("?" * len(manifest_chunk_ids))
        conn.execute(
            f"DELETE FROM chunks WHERE id IN ({placeholders})", manifest_chunk_ids
        )

    def delete_file_row(self, conn: sqlite3.Connection, file_id: bytes) -> None:
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

    def get_stale_file_ids(
        self, conn: sqlite3.Connection, active_source_ids: list[int]
    ) -> list[bytes]:
        placeholders = ",".join("?" * len(active_source_ids))
        cursor = conn.execute(
            f"SELECT id FROM files WHERE data_source_id NOT IN ({placeholders})",
            active_source_ids,
        )
        return [row[0] for row in cursor]

    def delete_stale_data_sources(
        self, conn: sqlite3.Connection, active_source_ids: list[int]
    ) -> None:
        placeholders = ",".join("?" * len(active_source_ids))
        conn.execute(
            f"DELETE FROM data_sources WHERE id NOT IN ({placeholders})",
            active_source_ids,
        )

    # ------------------------------------------------------------------
    # Source state (Google Drive page tokens, webhook expiry, etc.)
    # ------------------------------------------------------------------

    def get_source_state(self, data_source_id: int, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM source_state WHERE data_source_id = ? AND key = ?",
            (data_source_id, key),
        ).fetchone()
        return row[0] if row else None

    def set_source_state(self, data_source_id: int, key: str, value: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO source_state (data_source_id, key, value) VALUES (?, ?, ?)",
                (data_source_id, key, value),
            )

    # ------------------------------------------------------------------
    # Self-contained operations (run on the shared connection)
    # ------------------------------------------------------------------

    def transform_hash_changed(self, transform_hash: bytes) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM transform_hashes WHERE transform_hash = ?",
            (transform_hash,),
        ).fetchone()
        return row is None

    def register_source(self, source_type: str, source_identifier: str) -> int:
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO data_sources (source_type, source_identifier) VALUES (?, ?)",
                (source_type, source_identifier),
            )
            row = self._conn.execute(
                "SELECT id FROM data_sources WHERE source_type = ? AND source_identifier = ?",
                (source_type, source_identifier),
            ).fetchone()
        return row[0]

    def delete_stale_sources(self, active_ids: list[int]) -> None:
        if not active_ids:
            return
        placeholders = ",".join("?" * len(active_ids))
        with self._conn:
            self._conn.execute(
                f"DELETE FROM data_sources WHERE id NOT IN ({placeholders})",
                active_ids,
            )

    def get_memoized(self, file_id: bytes, fn_hash: bytes, args_hash: bytes) -> bytes | None:
        row = self._conn.execute(
            "SELECT result FROM memoize_cache WHERE file_id=? AND fn_hash=? AND args_hash=?",
            (file_id, fn_hash, args_hash),
        ).fetchone()
        return row[0] if row else None

    def set_memoized(self, file_id: bytes, fn_hash: bytes, args_hash: bytes, result: bytes) -> None:
        # `with self._conn` commits on exit. Committing synchronously here releases
        # the write before the transform's next `await`, so no write transaction is
        # ever held across a yield (the single-conn answer to "database is locked"
        # / commit-before-yield).
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO memoize_cache (file_id, fn_hash, args_hash, result) "
                "VALUES (?, ?, ?, ?)",
                (file_id, fn_hash, args_hash, result),
            )

    def sweep_memoized(
        self, file_id: bytes, temp_keys: set[tuple[bytes, bytes]]
    ) -> None:
        self._conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _sweep_keys (fn_hash BLOB, args_hash BLOB)"
        )
        # The connection is shared and long-lived, so this TEMP table persists
        # across files; clear the last file's keys before loading this file's.
        self._conn.execute("DELETE FROM _sweep_keys")
        with self._conn:
            self._conn.executemany("INSERT INTO _sweep_keys VALUES (?, ?)", temp_keys)
            self._conn.execute(
                "DELETE FROM memoize_cache "
                "WHERE file_id = ? "
                "AND (fn_hash, args_hash) NOT IN (SELECT fn_hash, args_hash FROM _sweep_keys)",
                (file_id,),
            )

    def update_transform_hash(self, transform_hash: bytes) -> None:
        """Replace the stored transform hash with the current one."""
        with self._conn:
            self._conn.execute("DELETE FROM transform_hashes")
            self._conn.execute(
                "INSERT INTO transform_hashes (transform_hash) VALUES (?)",
                (transform_hash,),
            )
