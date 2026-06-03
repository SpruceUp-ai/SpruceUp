import dataclasses
import json
import sqlite3
from contextlib import contextmanager

from .models import ChunkWrapper, SpruceFile

_MANIFEST_PATH = "spruceup_manifest.db"

_CONFIG_KEYS: frozenset[str] = frozenset({
    "file_cache_ready",
    "embedding_model",
})


class Manifest:
    """Single access point for all SQLite manifest reads and writes."""

    def __init__(self, path: str = _MANIFEST_PATH):
        self._path = path
        self._conn = sqlite3.connect(self._path, autocommit=True)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._init_db()

    # ------------------------------------------------------------------
    # Schema init
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self.transaction():
            self._init_sources_schema()
            self._init_files_schema()
            self._init_chunks_schema()
            self._init_transform_schema()
            self._init_memoize_schema()
            self._init_memoize_fn_hashes_schema()
            self._init_config_state_schema()

    def _init_sources_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS data_sources (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type       TEXT NOT NULL,
                source_identifier TEXT NOT NULL,
                UNIQUE(source_type, source_identifier)
            )
            """
        )
        self._conn.execute(
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

    def _init_files_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                id             BLOB PRIMARY KEY,
                source_ref     TEXT NOT NULL,
                content_hash   BLOB,
                data_source_id INTEGER REFERENCES data_sources(id) ON DELETE CASCADE,
                file_type      TEXT,
                raw_content    BLOB,
                sync_state     TEXT NOT NULL DEFAULT 'in_flight'
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS file_metadata (
                file_id BLOB NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                key     TEXT NOT NULL,
                value   TEXT NOT NULL,
                PRIMARY KEY (file_id, key)
            )
            """
        )

    def _init_chunks_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                file_id                BLOB NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                user_chunk_object_hash BLOB NOT NULL,
                user_chunk_object      BLOB NOT NULL,
                PRIMARY KEY (file_id, user_chunk_object_hash)
            )
            """
        )

    def _init_transform_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transform_hashes (
                transform_hash BLOB PRIMARY KEY
            )
            """
        )

    def _init_memoize_fn_hashes_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memoize_fn_hashes (
                fn_hash BLOB PRIMARY KEY
            )
            """
        )

    def _init_memoize_schema(self) -> None:
        self._conn.execute(
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

    def _init_config_state_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS config_state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

    # ------------------------------------------------------------------
    # Transaction context manager
    # ------------------------------------------------------------------

    @contextmanager
    def transaction(self):
        """Group multiple writes into one atomic transaction.

        Single writes outside this context auto-commit immediately via
        autocommit=True. Use this only when multiple writes must succeed or
        fail together.
        """
        self._conn.execute("BEGIN")
        try:
            yield
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    # ------------------------------------------------------------------
    # Chunk and file operations
    # ------------------------------------------------------------------

    def get_chunks_for_file(self, file_id: bytes) -> list[dict]:
        cursor = self._conn.execute(
            "SELECT user_chunk_object_hash FROM chunks WHERE file_id = ?",
            (file_id,),
        )
        return [{"content_hash": row[0]} for row in cursor]

    def upsert_chunks(self, chunks: list[tuple[bytes, ChunkWrapper]]) -> None:
        if not chunks:
            return
        rows = [
            (
                file_id,
                chunk.user_chunk_object_hash,
                json.dumps(dataclasses.asdict(chunk.user_chunk), default=str).encode(),
            )
            for file_id, chunk in chunks
        ]
        self._conn.executemany(
            """INSERT OR IGNORE INTO chunks
                   (file_id, user_chunk_object_hash, user_chunk_object)
               VALUES (?, ?, ?)""",
            rows,
        )

    def ensure_file_row_exists(self, file_id: bytes, source_ref: str) -> None:
        self._conn.execute(
            """INSERT INTO files (id, source_ref, sync_state)
               VALUES (?, ?, 'in_flight')
               ON CONFLICT (id) DO UPDATE SET
                   source_ref = excluded.source_ref,
                   sync_state = 'in_flight'""",
            (file_id, source_ref),
        )

    def upsert_file_row(self, file: SpruceFile) -> None:
        self._conn.execute(
            """INSERT INTO files
                   (id, source_ref, content_hash, data_source_id, file_type, raw_content)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT (id) DO UPDATE SET
                   source_ref     = excluded.source_ref,
                   content_hash   = excluded.content_hash,
                   data_source_id = excluded.data_source_id,
                   file_type      = excluded.file_type,
                   raw_content    = excluded.raw_content""",
            (
                file.file_id,
                file.source_ref,
                file.content_hash,
                file.data_source_id,
                file.file_type,
                file.raw_content if isinstance(file.raw_content, bytes) else file.raw_content.encode(),
            ),
        )

    def get_raw_content(self, file_id: bytes) -> bytes | None:
        row = self._conn.execute(
            "SELECT raw_content FROM files WHERE id = ?", (file_id,)
        ).fetchone()
        return row[0] if row else None

    def upsert_file_metadata(self, file_id: bytes, metadata: dict) -> None:
        if not metadata:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO file_metadata (file_id, key, value) VALUES (?, ?, ?)",
            [(file_id, k, str(v)) for k, v in metadata.items()],
        )

    def get_file_metadata(self, file_id: bytes) -> dict:
        rows = self._conn.execute(
            "SELECT key, value FROM file_metadata WHERE file_id = ?",
            (file_id,),
        ).fetchall()
        return {key: value for key, value in rows}

    def get_source_refs(self, data_source_id: int) -> set[str]:
        rows = self._conn.execute(
            "SELECT source_ref FROM files WHERE data_source_id = ?",
            (data_source_id,),
        ).fetchall()
        return {row[0] for row in rows}

    def get_files_with_metadata(self, data_source_id: int) -> list[dict]:
        rows = self._conn.execute(
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

    def get_file_id_by_ref(self, source_ref: str) -> bytes | None:
        row = self._conn.execute(
            "SELECT id FROM files WHERE source_ref = ?", (source_ref,)
        ).fetchone()
        return row[0] if row else None

    def update_file_ref(self, file_id: bytes, new_ref: str) -> None:
        self._conn.execute(
            "UPDATE files SET source_ref = ? WHERE id = ?", (new_ref, file_id)
        )

    def chunk_hash_referenced_elsewhere(
        self,
        content_hash: bytes,
        exclude_file_ids: list[bytes],
    ) -> bool:
        placeholders = ",".join("?" * len(exclude_file_ids))
        return self._conn.execute(
            f"SELECT 1 FROM chunks WHERE user_chunk_object_hash = ? AND file_id NOT IN ({placeholders})",
            [content_hash, *exclude_file_ids],
        ).fetchone() is not None

    def delete_chunks(self, chunk_keys: list[tuple[bytes, bytes]]) -> None:
        if not chunk_keys:
            return
        self._conn.executemany(
            "DELETE FROM chunks WHERE file_id = ? AND user_chunk_object_hash = ?",
            chunk_keys,
        )

    def delete_file_row(self, file_id: bytes) -> None:
        self._conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

    def get_orphaned_file_ids(self, active_source_ids: list[int]) -> list[bytes]:
        placeholders = ",".join("?" * len(active_source_ids))
        cursor = self._conn.execute(
            f"SELECT id FROM files WHERE data_source_id NOT IN ({placeholders})",
            active_source_ids,
        )
        return [row[0] for row in cursor]

    def purge_inactive_sources(self, active_source_ids: list[int]) -> None:
        placeholders = ",".join("?" * len(active_source_ids))
        self._conn.execute(
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
        self._conn.execute(
            "INSERT OR REPLACE INTO source_state (data_source_id, key, value) VALUES (?, ?, ?)",
            (data_source_id, key, value),
        )

    # ------------------------------------------------------------------
    # Self-contained operations
    # ------------------------------------------------------------------

    def transform_hash_changed(self, transform_hash: bytes) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM transform_hashes WHERE transform_hash = ?",
            (transform_hash,),
        ).fetchone()
        return row is None

    def any_memoize_fn_hash_missing(self, fn_hashes: set[bytes]) -> bool:
        if not fn_hashes:
            return False
        placeholders = ",".join("?" * len(fn_hashes))
        found = self._conn.execute(
            f"SELECT COUNT(*) FROM memoize_fn_hashes WHERE fn_hash IN ({placeholders})",
            list(fn_hashes),
        ).fetchone()[0]
        return found < len(fn_hashes)

    def reset_in_flight_to_failed(self) -> None:
        self._conn.execute(
            "UPDATE files SET sync_state = 'failed' WHERE sync_state = 'in_flight'"
        )

    def set_sync_state(self, file_id: bytes, state: str) -> None:
        self._conn.execute(
            "UPDATE files SET sync_state = ? WHERE id = ?", (state, file_id)
        )

    def get_failed_files(self, data_source_id: int) -> list[tuple[str, int]]:
        rows = self._conn.execute(
            "SELECT source_ref, data_source_id FROM files "
            "WHERE data_source_id = ? AND sync_state = 'failed'",
            (data_source_id,),
        ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def register_source(self, source_type: str, source_identifier: str) -> int:
        self._conn.execute(
            "INSERT OR IGNORE INTO data_sources (source_type, source_identifier) VALUES (?, ?)",
            (source_type, source_identifier),
        )
        row = self._conn.execute(
            "SELECT id FROM data_sources WHERE source_type = ? AND source_identifier = ?",
            (source_type, source_identifier),
        ).fetchone()
        return row[0]

    def get_memoized(self, file_id: bytes, fn_hash: bytes, args_hash: bytes) -> bytes | None:
        row = self._conn.execute(
            "SELECT result FROM memoize_cache WHERE file_id=? AND fn_hash=? AND args_hash=?",
            (file_id, fn_hash, args_hash),
        ).fetchone()
        return row[0] if row else None

    def set_memoized(self, file_id: bytes, fn_hash: bytes, args_hash: bytes, result: bytes) -> None:
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
        self._conn.execute("DELETE FROM _sweep_keys")
        with self.transaction():
            self._conn.executemany("INSERT INTO _sweep_keys VALUES (?, ?)", temp_keys)
            self._conn.execute(
                "DELETE FROM memoize_cache "
                "WHERE file_id = ? "
                "AND (fn_hash, args_hash) NOT IN (SELECT fn_hash, args_hash FROM _sweep_keys)",
                (file_id,),
            )

    def update_transform_hash(self, transform_hash: bytes) -> None:
        with self.transaction():
            self._conn.execute("DELETE FROM transform_hashes")
            self._conn.execute(
                "INSERT INTO transform_hashes (transform_hash) VALUES (?)",
                (transform_hash,),
            )

    def update_memoize_fn_hashes(self, fn_hashes: set[bytes]) -> None:
        with self.transaction():
            self._conn.execute("DELETE FROM memoize_fn_hashes")
            self._conn.executemany(
                "INSERT INTO memoize_fn_hashes (fn_hash) VALUES (?)",
                [(h,) for h in fn_hashes],
            )

    # ------------------------------------------------------------------
    # Config state
    # ------------------------------------------------------------------

    def get_config_value(self, key: str) -> str | None:
        if key not in _CONFIG_KEYS:
            raise ValueError(f"Unknown config key: {key!r}")
        row = self._conn.execute(
            "SELECT value FROM config_state WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set_config_value(self, key: str, value: str) -> None:
        if key not in _CONFIG_KEYS:
            raise ValueError(f"Unknown config key: {key!r}")
        self._conn.execute(
            "INSERT OR REPLACE INTO config_state (key, value) VALUES (?, ?)",
            (key, value),
        )
