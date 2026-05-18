"""
Tests for SyncEngine.

Postgres is mocked — no real DB needed.
The SQLite manifest runs against a real temp file so manifest reads
and writes are genuinely exercised.
"""

import sqlite3
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from db import init_db
from models import UserDefinedChunkSchema
from sync_engine import (
    ChunkWrapper,
    SpruceFile,
    SyncEngine,
    hash_chunk_id,
    hash_file_path,
    hash_object,
)


# ---------------------------------------------------------------------------
# Minimal test schema and Postgres mock
# ---------------------------------------------------------------------------

@dataclass
class SimpleChunkSchema(UserDefinedChunkSchema):
    """Uses only the three base fields: id, chunk_text, chunk_embedding."""
    pass


class MockPgConn:
    """Records SQL calls instead of sending them to Postgres."""

    def __init__(self):
        self.calls: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, params=None):
        self.calls.append({"sql": sql.strip(), "params": params})

    def executemany(self, sql, rows):
        self.calls.append({"sql": sql.strip(), "rows": list(rows)})

    def inserted_ids(self) -> list:
        """First column (PK) of every row sent to INSERT executemany calls."""
        return [
            row[0]
            for call in self.calls if "INSERT" in call["sql"]
            for row in call.get("rows", [])
        ]

    def deleted_ids(self) -> list:
        """All params passed to DELETE execute calls, flattened into one list."""
        return [
            pk
            for call in self.calls if "DELETE" in call["sql"]
            for pk in (call["params"] or [])
        ]

    def reset(self) -> None:
        self.calls.clear()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FILE_PATH_A = "corpus/doc_a.pdf"
FILE_PATH_B = "corpus/doc_b.md"
FILE_ID_A = hash_file_path(FILE_PATH_A)
FILE_ID_B = hash_file_path(FILE_PATH_B)


@pytest.fixture
def pg():
    conn = MockPgConn()
    with patch("psycopg.connect", return_value=conn):
        yield conn


@pytest.fixture
def tmp_manifest(tmp_path):
    return str(tmp_path / "manifest.db")


@pytest.fixture
def engine(tmp_manifest, pg):
    init_db(tmp_manifest)
    e = SyncEngine(manifest_path=tmp_manifest, pg_connstr="dbname=test")
    e.define_target_table(
        db_name="test",
        table_name="vectors",
        schema_from_class=SimpleChunkSchema,
        primary_key="id",
    )
    return e


# ---------------------------------------------------------------------------
# Helper constructors and manifest query utilities
# ---------------------------------------------------------------------------

def make_chunk(file_path: str, chunk_id: str, text: str, ordinal: int) -> ChunkWrapper:
    user_chunk = SimpleChunkSchema(id=chunk_id, chunk_text=text, chunk_embedding=[0.1, 0.2, 0.3])
    return ChunkWrapper(
        user_chunk=user_chunk,
        user_chunk_object_hash=hash_object(user_chunk),
        ordinal=ordinal,
        chunk_id=hash_chunk_id(file_path, ordinal),
    )


def make_file(file_path: str, chunks: list[ChunkWrapper], mtime: float = 1_000_000.0) -> SpruceFile:
    fid = hash_file_path(file_path)
    return SpruceFile(
        file_id=fid,
        file_path=file_path,
        inode=0,
        mtime=mtime,
        content_hash=fid,
        transform_hash=fid,
        file_type=file_path.rsplit(".", 1)[-1],
        data_source_id=1,
        raw_content=b"",
        parsed_content=None,
        chunk_strs=[],
        chunks=chunks,
    )


def chunk_count(manifest_path: str, file_id: bytes | None = None) -> int:
    with sqlite3.connect(manifest_path) as conn:
        if file_id is not None:
            return conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE file_id = ?", (file_id,)
            ).fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def file_row(manifest_path: str, file_id: bytes) -> dict | None:
    with sqlite3.connect(manifest_path) as conn:
        row = conn.execute(
            "SELECT id, mtime FROM files WHERE id = ?", (file_id,)
        ).fetchone()
    return {"id": row[0], "mtime": row[1]} if row else None


# ---------------------------------------------------------------------------
# reconcile tests
# ---------------------------------------------------------------------------

class TestReconcile:

    def test_requires_define_target_table(self, tmp_manifest, pg):
        init_db(tmp_manifest)
        engine = SyncEngine(manifest_path=tmp_manifest, pg_connstr="dbname=test")
        with pytest.raises(AssertionError):
            engine.reconcile([])

    def test_new_chunks_upserted_to_postgres(self, engine, pg):
        chunks = [
            make_chunk(FILE_PATH_A, "c1", "First chunk", ordinal=1),
            make_chunk(FILE_PATH_A, "c2", "Second chunk", ordinal=2),
        ]
        engine.reconcile([make_file(FILE_PATH_A, chunks)])
        assert set(pg.inserted_ids()) == {"c1", "c2"}

    def test_new_chunks_written_to_manifest(self, engine, tmp_manifest):
        chunks = [
            make_chunk(FILE_PATH_A, "c1", "First chunk", ordinal=1),
            make_chunk(FILE_PATH_A, "c2", "Second chunk", ordinal=2),
        ]
        engine.reconcile([make_file(FILE_PATH_A, chunks)])
        assert chunk_count(tmp_manifest, FILE_ID_A) == 2

    def test_file_row_written_to_manifest(self, engine, tmp_manifest):
        chunks = [make_chunk(FILE_PATH_A, "c1", "First chunk", ordinal=1)]
        engine.reconcile([make_file(FILE_PATH_A, chunks, mtime=1_234_567.0)])
        row = file_row(tmp_manifest, FILE_ID_A)
        assert row is not None
        assert row["mtime"] == 1_234_567.0

    def test_unchanged_chunks_not_reupserted(self, engine, pg):
        chunks = [make_chunk(FILE_PATH_A, "c1", "Same text", ordinal=1)]
        engine.reconcile([make_file(FILE_PATH_A, chunks)])
        pg.reset()
        engine.reconcile([make_file(FILE_PATH_A, chunks)])
        assert pg.inserted_ids() == []

    def test_changed_chunk_is_upserted(self, engine, pg):
        v1 = [
            make_chunk(FILE_PATH_A, "c1", "Original text", ordinal=1),
            make_chunk(FILE_PATH_A, "c2", "Unchanged text", ordinal=2),
        ]
        engine.reconcile([make_file(FILE_PATH_A, v1)])
        pg.reset()

        v2 = [
            make_chunk(FILE_PATH_A, "c1", "EDITED text", ordinal=1),
            make_chunk(FILE_PATH_A, "c2", "Unchanged text", ordinal=2),
        ]
        engine.reconcile([make_file(FILE_PATH_A, v2)])
        assert pg.inserted_ids() == ["c1"]

    def test_orphaned_chunk_deleted_from_postgres(self, engine, pg):
        v1 = [
            make_chunk(FILE_PATH_A, "c1", "Chunk one", ordinal=1),
            make_chunk(FILE_PATH_A, "c2", "Chunk two", ordinal=2),
        ]
        engine.reconcile([make_file(FILE_PATH_A, v1)])
        pg.reset()

        v2 = [make_chunk(FILE_PATH_A, "c1", "Chunk one", ordinal=1)]
        engine.reconcile([make_file(FILE_PATH_A, v2)])
        assert "c2" in pg.deleted_ids()

    def test_orphaned_chunk_removed_from_manifest(self, engine, tmp_manifest):
        v1 = [
            make_chunk(FILE_PATH_A, "c1", "Chunk one", ordinal=1),
            make_chunk(FILE_PATH_A, "c2", "Chunk two", ordinal=2),
        ]
        engine.reconcile([make_file(FILE_PATH_A, v1)])
        assert chunk_count(tmp_manifest, FILE_ID_A) == 2

        v2 = [make_chunk(FILE_PATH_A, "c1", "Chunk one", ordinal=1)]
        engine.reconcile([make_file(FILE_PATH_A, v2)])
        assert chunk_count(tmp_manifest, FILE_ID_A) == 1

    def test_file_row_mtime_updated_on_second_reconcile(self, engine, tmp_manifest):
        chunks = [make_chunk(FILE_PATH_A, "c1", "Same text", ordinal=1)]
        engine.reconcile([make_file(FILE_PATH_A, chunks, mtime=1_000_000.0)])
        engine.reconcile([make_file(FILE_PATH_A, chunks, mtime=2_000_000.0)])
        assert file_row(tmp_manifest, FILE_ID_A)["mtime"] == 2_000_000.0

    def test_multiple_files_in_one_call(self, engine, tmp_manifest):
        engine.reconcile([
            make_file(FILE_PATH_A, [make_chunk(FILE_PATH_A, "a1", "Doc A", ordinal=1)]),
            make_file(FILE_PATH_B, [
                make_chunk(FILE_PATH_B, "b1", "Doc B chunk one", ordinal=1),
                make_chunk(FILE_PATH_B, "b2", "Doc B chunk two", ordinal=2),
            ]),
        ])
        assert chunk_count(tmp_manifest, FILE_ID_A) == 1
        assert chunk_count(tmp_manifest, FILE_ID_B) == 2
        assert file_row(tmp_manifest, FILE_ID_A) is not None
        assert file_row(tmp_manifest, FILE_ID_B) is not None

    def test_reconcile_only_touches_given_files(self, engine, tmp_manifest):
        engine.reconcile([
            make_file(FILE_PATH_A, [make_chunk(FILE_PATH_A, "a1", "Doc A", ordinal=1)]),
            make_file(FILE_PATH_B, [make_chunk(FILE_PATH_B, "b1", "Doc B", ordinal=1)]),
        ])
        # Reconcile only file B — file A's chunks must remain untouched
        engine.reconcile([
            make_file(FILE_PATH_B, [make_chunk(FILE_PATH_B, "b1", "Doc B", ordinal=1)])
        ])
        assert chunk_count(tmp_manifest, FILE_ID_A) == 1


# ---------------------------------------------------------------------------
# delete_file tests
# ---------------------------------------------------------------------------

class TestDeleteFile:

    def test_requires_define_target_table(self, tmp_manifest, pg):
        init_db(tmp_manifest)
        engine = SyncEngine(manifest_path=tmp_manifest, pg_connstr="dbname=test")
        with pytest.raises(AssertionError):
            engine.delete_file(FILE_ID_A)

    def test_sends_chunk_pks_to_postgres(self, engine, pg):
        chunks = [
            make_chunk(FILE_PATH_A, "c1", "Chunk one", ordinal=1),
            make_chunk(FILE_PATH_A, "c2", "Chunk two", ordinal=2),
        ]
        engine.reconcile([make_file(FILE_PATH_A, chunks)])
        pg.reset()

        engine.delete_file(FILE_ID_A)
        assert set(pg.deleted_ids()) == {"c1", "c2"}

    def test_removes_chunks_from_manifest(self, engine, tmp_manifest):
        chunks = [
            make_chunk(FILE_PATH_A, "c1", "Chunk one", ordinal=1),
            make_chunk(FILE_PATH_A, "c2", "Chunk two", ordinal=2),
        ]
        engine.reconcile([make_file(FILE_PATH_A, chunks)])
        engine.delete_file(FILE_ID_A)
        assert chunk_count(tmp_manifest, FILE_ID_A) == 0

    def test_removes_file_row_from_manifest(self, engine, tmp_manifest):
        chunks = [make_chunk(FILE_PATH_A, "c1", "Chunk one", ordinal=1)]
        engine.reconcile([make_file(FILE_PATH_A, chunks)])
        engine.delete_file(FILE_ID_A)
        assert file_row(tmp_manifest, FILE_ID_A) is None

    def test_unknown_file_does_not_call_postgres(self, engine, pg):
        pg.reset()
        engine.delete_file(FILE_ID_A)  # never reconciled, no chunks in manifest
        assert pg.deleted_ids() == []
