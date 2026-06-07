import sqlite3
import struct
from contextlib import contextmanager

from .models import ChunkWrapper, SpruceFile

_MANIFEST_PATH = "spruceup_manifest.db"


def _pack_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}d", *embedding)


def _unpack_embedding(blob: bytes) -> list[float]:
    n = len(blob) // 8
    return list(struct.unpack(f"{n}d", blob))

_CONFIG_KEYS: frozenset[str] = frozenset({
    "file_cache_ready",
    "embedding_model",
    "embedding_dimensions",
    "target_identity",
    "schema_fingerprint",
})


class Manifest:
    """Single access point for all SQLite manifest reads and writes."""

    def __init__(self, path: str = _MANIFEST_PATH):
        self.path = path
        self._conn = sqlite3.connect(self.path, autocommit=True)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._init_db()

    def close(self) -> None:
        self._conn.close()

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
            self._init_embedding_cache_schema()

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
                id                  TEXT PRIMARY KEY,
                content_hash        BLOB,
                data_source_id      INTEGER REFERENCES data_sources(id) ON DELETE CASCADE,
                file_type           TEXT,
                raw_content         BLOB,
                modified_at         REAL,
                sync_state          TEXT NOT NULL DEFAULT 'in_flight',
                last_change_type    TEXT
            )
            """
        )

    def _init_chunks_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                file_id                TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                user_chunk_object_hash BLOB NOT NULL,
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
                file_id   TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                fn_hash   BLOB NOT NULL,
                args_hash BLOB NOT NULL,
                result    BLOB NOT NULL,
                PRIMARY KEY (file_id, fn_hash, args_hash)
            )
            """
        )

    def _init_embedding_cache_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                file_id         TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                chunk_text_hash BLOB NOT NULL,
                embedding       BLOB NOT NULL,
                PRIMARY KEY (file_id, chunk_text_hash)
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

    def get_chunks_for_file(self, file_id: str) -> list[dict]:
        cursor = self._conn.execute(
            "SELECT user_chunk_object_hash FROM chunks WHERE file_id = ?",
            (file_id,),
        )
        return [{"user_chunk_object_hash": row[0]} for row in cursor]

    def upsert_chunks(self, chunks: list[tuple[str, ChunkWrapper]]) -> None:
        if not chunks:
            return
        self._conn.executemany(
            "INSERT OR IGNORE INTO chunks (file_id, user_chunk_object_hash) VALUES (?, ?)",
            [(file_id, chunk.user_chunk_object_hash) for file_id, chunk in chunks],
        )

    def ensure_file_row_exists(self, file_id: str, data_source_id: int) -> None:
        # FK placeholder so per-file cache writes during transform don't violate
        # the files(id) foreign key. data_source_id is set now (not deferred to
        # upsert_file_row) so a crash before reconcile leaves a row the sweeper
        # can still resolve, rather than one with a NULL source.
        self._conn.execute(
            """INSERT INTO files (id, data_source_id, sync_state)
               VALUES (?, ?, 'in_flight')
               ON CONFLICT (id) DO UPDATE SET
                   data_source_id = excluded.data_source_id,
                   sync_state = 'in_flight'""",
            (file_id, data_source_id),
        )

    def upsert_file_row(self, file: SpruceFile) -> None:
        self._conn.execute(
            """INSERT INTO files
                   (id, content_hash, data_source_id, file_type, raw_content, modified_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT (id) DO UPDATE SET
                   content_hash   = excluded.content_hash,
                   data_source_id = excluded.data_source_id,
                   file_type      = excluded.file_type,
                   raw_content    = excluded.raw_content,
                   modified_at    = excluded.modified_at""",
            (
                file.file_id,
                file.content_hash,
                file.data_source_id,
                file.file_type,
                file.raw_content if isinstance(file.raw_content, bytes) else file.raw_content.encode(),
                file.modified_at,
            ),
        )

    def get_raw_content(self, file_id: str) -> bytes | None:
        row = self._conn.execute(
            "SELECT raw_content FROM files WHERE id = ?", (file_id,)
        ).fetchone()
        return row[0] if row else None

    def get_file_modified_at(self, file_id: str) -> float | None:
        row = self._conn.execute(
            "SELECT modified_at FROM files WHERE id = ?", (file_id,)
        ).fetchone()
        return row[0] if row else None

    def get_files_for_source(self, data_source_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, content_hash, modified_at FROM files WHERE data_source_id = ?",
            (data_source_id,),
        ).fetchall()
        return [
            {"file_id": row[0], "content_hash": row[1], "modified_at": row[2]}
            for row in rows
        ]

    def delete_chunks(self, chunk_keys: list[tuple[str, bytes]]) -> None:
        if not chunk_keys:
            return
        self._conn.executemany(
            "DELETE FROM chunks WHERE file_id = ? AND user_chunk_object_hash = ?",
            chunk_keys,
        )

    def delete_file_row(self, file_id: str) -> None:
        self._conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

    def get_orphaned_files(self, active_source_ids: list[int]) -> list[dict]:
        placeholders = ",".join("?" * len(active_source_ids))
        cursor = self._conn.execute(
            f"SELECT id, data_source_id FROM files WHERE data_source_id NOT IN ({placeholders})",
            active_source_ids,
        )
        return [{"file_id": row[0], "data_source_id": row[1]} for row in cursor]

    def purge_empty_inactive_sources(self, active_source_ids: list[int]) -> None:
        # Drop inactive sources that no longer have any files. A source whose
        # delete is still pending keeps its files (and so its row) until the
        # sweeper drains them, then it's purged on a later startup.
        placeholders = ",".join("?" * len(active_source_ids))
        self._conn.execute(
            f"DELETE FROM data_sources "
            f"WHERE id NOT IN ({placeholders}) "
            f"AND id NOT IN ("
            f"SELECT data_source_id FROM files WHERE data_source_id IS NOT NULL"
            f")",
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

    def set_sync_state(self, file_id: str, state: str) -> None:
        self._conn.execute(
            "UPDATE files SET sync_state = ? WHERE id = ?", (state, file_id)
        )

    def mark_failed(self, file_id: str, change_type: str) -> None:
        self._conn.execute(
            """UPDATE files
               SET sync_state = 'failed',
                   last_change_type = ?
               WHERE id = ?""",
            (change_type, file_id),
        )

    def get_failed_files(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, data_source_id, last_change_type "
            "FROM files WHERE sync_state = 'failed'",
        ).fetchall()
        return [
            {
                "file_id": row[0],
                "data_source_id": row[1],
                "change_type": row[2],
            }
            for row in rows
        ]

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

    def get_memoized(self, file_id: str, fn_hash: bytes, args_hash: bytes) -> bytes | None:
        row = self._conn.execute(
            "SELECT result FROM memoize_cache WHERE file_id=? AND fn_hash=? AND args_hash=?",
            (file_id, fn_hash, args_hash),
        ).fetchone()
        return row[0] if row else None

    def set_memoized(self, file_id: str, fn_hash: bytes, args_hash: bytes, result: bytes) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO memoize_cache (file_id, fn_hash, args_hash, result) "
            "VALUES (?, ?, ?, ?)",
            (file_id, fn_hash, args_hash, result),
        )

    def sweep_memoized(
        self, file_id: str, temp_keys: set[tuple[bytes, bytes]]
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

    def get_cached_embeddings(
        self, file_id: str, chunk_text_hashes: list[bytes]
    ) -> dict[bytes, list[float]]:
        if not chunk_text_hashes:
            return {}
        placeholders = ",".join("?" * len(chunk_text_hashes))
        rows = self._conn.execute(
            f"SELECT chunk_text_hash, embedding FROM embedding_cache "
            f"WHERE file_id = ? AND chunk_text_hash IN ({placeholders})",
            [file_id, *chunk_text_hashes],
        ).fetchall()
        return {row[0]: _unpack_embedding(row[1]) for row in rows}

    def set_cached_embeddings(
        self, file_id: str, entries: list[tuple[bytes, list[float]]]
    ) -> None:
        if not entries:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO embedding_cache (file_id, chunk_text_hash, embedding) "
            "VALUES (?, ?, ?)",
            [(file_id, h, _pack_embedding(e)) for h, e in entries],
        )

    def sweep_embedding_cache(self, file_id: str, used_hashes: set[bytes]) -> None:
        if not used_hashes:
            self._conn.execute(
                "DELETE FROM embedding_cache WHERE file_id = ?", (file_id,)
            )
            return
        placeholders = ",".join("?" * len(used_hashes))
        self._conn.execute(
            f"DELETE FROM embedding_cache WHERE file_id = ? "
            f"AND chunk_text_hash NOT IN ({placeholders})",
            [file_id, *used_hashes],
        )

    def flush_embedding_cache(self) -> None:
        self._conn.execute("DELETE FROM embedding_cache")

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
