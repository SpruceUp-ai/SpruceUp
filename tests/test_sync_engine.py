"""
Tests for SyncEngine.

Postgres is mocked — no real DB needed.
The SQLite manifest runs against a real temp file so manifest reads
and writes are genuinely exercised.
"""

import sqlite3
from dataclasses import dataclass

import pytest

from spruceup.manifest import Manifest
from spruceup.models import ChunkWrapper
from spruceup.sync_engine import (
    ChunkWrapper,
    SpruceFile,
    SyncEngine,
    hash_chunk_id,
    hash_file_path,
    hash_object,
)
from spruceup.connectors.base import TargetConnector


# ---------------------------------------------------------------------------
# Minimal test schema and TargetConnector mock
# ---------------------------------------------------------------------------

@dataclass
class SimpleChunk:
    id: str
    chunk_text: str
    chunk_embedding: list[float]


class MockSyncTarget(TargetConnector):
    """Records sync calls instead of writing to a real database."""

    primary_key = "id"

    def __init__(self):
        self.calls: list[dict] = []

    @property
    def display_name(self) -> str:
        return "mock_target"

    def ensure_table_exists(self, embedding_dimensions: int) -> None:
        pass

    def sync(self, upserts: list[ChunkWrapper], deletes: list) -> None:
        self.calls.append({"upserts": list(upserts), "deletes": list(deletes)})

    def inserted_ids(self) -> list:
        return [chunk.user_chunk.id for call in self.calls for chunk in call["upserts"]]

    def deleted_ids(self) -> list:
        return [pk for call in self.calls for pk in call["deletes"]]

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
    return MockSyncTarget()


@pytest.fixture
def tmp_manifest(tmp_path):
    return str(tmp_path / "manifest.db")


@pytest.fixture
def engine(tmp_manifest, pg):
    manifest = Manifest(tmp_manifest)
    manifest.register_source("local", "/test-corpus")  # creates data_sources row id=1
    return SyncEngine(manifest=manifest, target=pg)


# ---------------------------------------------------------------------------
# Helper constructors and manifest query utilities
# ---------------------------------------------------------------------------

def make_chunk(file_path: str, chunk_id: str, text: str, ordinal: int) -> ChunkWrapper:
    user_chunk = SimpleChunk(id=chunk_id, chunk_text=text, chunk_embedding=[0.1, 0.2, 0.3])
    return ChunkWrapper(
        user_chunk=user_chunk,
        user_chunk_object_hash=hash_object(user_chunk),
        ordinal=ordinal,
        chunk_id=hash_chunk_id(file_path, ordinal),
    )


def make_file(
    file_path: str,
    chunks: list[ChunkWrapper],
    mtime: float = 1_000_000.0,
    data_source_id: int = 1,
) -> SpruceFile:
    fid = hash_file_path(file_path)
    return SpruceFile(
        file_id=fid,
        file_path=file_path,
        inode=0,
        mtime=mtime,
        content_hash=fid,
        file_type=file_path.rsplit(".", 1)[-1],
        data_source_id=data_source_id,
        raw_content=b"",
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


def file_path_in_manifest(manifest_path: str, file_id: bytes) -> str | None:
    with sqlite3.connect(manifest_path) as conn:
        row = conn.execute(
            "SELECT file_path FROM files WHERE id = ?", (file_id,)
        ).fetchone()
    return row[0] if row else None


def data_source_exists(manifest_path: str, source_id: int) -> bool:
    with sqlite3.connect(manifest_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM data_sources WHERE id = ?", (source_id,)
        ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# reconcile tests
# ---------------------------------------------------------------------------

class TestReconcile:

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

    def test_empty_file_list_is_noop(self, engine, pg):
        engine.reconcile([])
        assert pg.inserted_ids() == []
        assert pg.deleted_ids() == []

    def test_file_with_no_chunks_writes_file_row(self, engine, tmp_manifest):
        engine.reconcile([make_file(FILE_PATH_A, [])])
        assert file_row(tmp_manifest, FILE_ID_A) is not None
        assert chunk_count(tmp_manifest, FILE_ID_A) == 0

    def test_all_chunks_removed_on_second_reconcile(self, engine, pg, tmp_manifest):
        v1 = [
            make_chunk(FILE_PATH_A, "c1", "Chunk one", ordinal=1),
            make_chunk(FILE_PATH_A, "c2", "Chunk two", ordinal=2),
        ]
        engine.reconcile([make_file(FILE_PATH_A, v1)])
        pg.reset()

        engine.reconcile([make_file(FILE_PATH_A, [])])
        assert set(pg.deleted_ids()) == {"c1", "c2"}
        assert chunk_count(tmp_manifest, FILE_ID_A) == 0


# ---------------------------------------------------------------------------
# delete_file tests
# ---------------------------------------------------------------------------

class TestDeleteFile:
    pytestmark = pytest.mark.anyio

    async def test_sends_chunk_pks_to_postgres(self, engine, pg):
        chunks = [
            make_chunk(FILE_PATH_A, "c1", "Chunk one", ordinal=1),
            make_chunk(FILE_PATH_A, "c2", "Chunk two", ordinal=2),
        ]
        engine.reconcile([make_file(FILE_PATH_A, chunks)])
        pg.reset()

        await engine.delete_file(FILE_PATH_A)
        assert set(pg.deleted_ids()) == {"c1", "c2"}

    async def test_removes_chunks_from_manifest(self, engine, tmp_manifest):
        chunks = [
            make_chunk(FILE_PATH_A, "c1", "Chunk one", ordinal=1),
            make_chunk(FILE_PATH_A, "c2", "Chunk two", ordinal=2),
        ]
        engine.reconcile([make_file(FILE_PATH_A, chunks)])
        await engine.delete_file(FILE_PATH_A)
        assert chunk_count(tmp_manifest, FILE_ID_A) == 0

    async def test_removes_file_row_from_manifest(self, engine, tmp_manifest):
        chunks = [make_chunk(FILE_PATH_A, "c1", "Chunk one", ordinal=1)]
        engine.reconcile([make_file(FILE_PATH_A, chunks)])
        await engine.delete_file(FILE_PATH_A)
        assert file_row(tmp_manifest, FILE_ID_A) is None

    async def test_unknown_file_does_not_delete_from_postgres(self, engine, pg):
        pg.reset()
        await engine.delete_file(FILE_PATH_A)  # never reconciled, no chunks in manifest
        assert pg.deleted_ids() == []


# ---------------------------------------------------------------------------
# move_file tests
# ---------------------------------------------------------------------------

class TestMoveFile:
    pytestmark = pytest.mark.anyio

    async def test_old_file_row_removed(self, engine, tmp_manifest):
        engine.reconcile([make_file(FILE_PATH_A, [make_chunk(FILE_PATH_A, "c1", "text", ordinal=1)])])
        await engine.move_file(FILE_PATH_A, FILE_PATH_B)
        assert file_row(tmp_manifest, FILE_ID_A) is None

    async def test_new_file_row_created(self, engine, tmp_manifest):
        engine.reconcile([make_file(FILE_PATH_A, [make_chunk(FILE_PATH_A, "c1", "text", ordinal=1)])])
        await engine.move_file(FILE_PATH_A, FILE_PATH_B)
        assert file_row(tmp_manifest, FILE_ID_B) is not None

    async def test_new_file_row_has_correct_path(self, engine, tmp_manifest):
        engine.reconcile([make_file(FILE_PATH_A, [make_chunk(FILE_PATH_A, "c1", "text", ordinal=1)])])
        await engine.move_file(FILE_PATH_A, FILE_PATH_B)
        assert file_path_in_manifest(tmp_manifest, FILE_ID_B) == FILE_PATH_B

    async def test_chunks_repointed_to_new_file(self, engine, tmp_manifest):
        chunks = [
            make_chunk(FILE_PATH_A, "c1", "Chunk one", ordinal=1),
            make_chunk(FILE_PATH_A, "c2", "Chunk two", ordinal=2),
        ]
        engine.reconcile([make_file(FILE_PATH_A, chunks)])
        await engine.move_file(FILE_PATH_A, FILE_PATH_B)
        assert chunk_count(tmp_manifest, FILE_ID_A) == 0
        assert chunk_count(tmp_manifest, FILE_ID_B) == 2

    async def test_does_not_write_to_postgres(self, engine, pg):
        engine.reconcile([make_file(FILE_PATH_A, [make_chunk(FILE_PATH_A, "c1", "text", ordinal=1)])])
        pg.reset()
        await engine.move_file(FILE_PATH_A, FILE_PATH_B)
        assert pg.calls == []

    async def test_unknown_file_move_is_noop(self, engine, tmp_manifest):
        await engine.move_file(FILE_PATH_A, FILE_PATH_B)
        assert file_row(tmp_manifest, FILE_ID_A) is None
        assert file_row(tmp_manifest, FILE_ID_B) is None


# ---------------------------------------------------------------------------
# delete_stale_sources tests
# ---------------------------------------------------------------------------

class TestDeleteStaleSources:

    def _seed_two_sources(self, engine):
        # Source 1 already registered by the engine fixture.
        stale_source_id = engine._manifest.register_source("local", "/old-corpus")
        engine.reconcile([
            make_file(FILE_PATH_A, [
                make_chunk(FILE_PATH_A, "a1", "active", ordinal=1),
            ], data_source_id=1),
            make_file(FILE_PATH_B, [
                make_chunk(FILE_PATH_B, "s1", "stale-one", ordinal=1),
                make_chunk(FILE_PATH_B, "s2", "stale-two", ordinal=2),
            ], data_source_id=stale_source_id),
        ])
        return stale_source_id

    def test_stale_chunks_sent_to_target_as_deletes(self, engine, pg):
        self._seed_two_sources(engine)
        pg.reset()

        engine.delete_stale_sources(active_ids=[1])
        assert set(pg.deleted_ids()) == {"s1", "s2"}

    def test_active_chunks_not_sent_as_deletes(self, engine, pg):
        self._seed_two_sources(engine)
        pg.reset()

        engine.delete_stale_sources(active_ids=[1])
        assert "a1" not in pg.deleted_ids()

    def test_stale_data_source_row_removed(self, engine, tmp_manifest):
        stale_id = self._seed_two_sources(engine)
        engine.delete_stale_sources(active_ids=[1])
        assert not data_source_exists(tmp_manifest, stale_id)

    def test_stale_file_row_cascade_removed(self, engine, tmp_manifest):
        self._seed_two_sources(engine)
        engine.delete_stale_sources(active_ids=[1])
        assert file_row(tmp_manifest, FILE_ID_B) is None

    def test_stale_chunks_cascade_removed(self, engine, tmp_manifest):
        self._seed_two_sources(engine)
        engine.delete_stale_sources(active_ids=[1])
        assert chunk_count(tmp_manifest, FILE_ID_B) == 0

    def test_active_source_rows_preserved(self, engine, tmp_manifest):
        self._seed_two_sources(engine)
        engine.delete_stale_sources(active_ids=[1])
        assert data_source_exists(tmp_manifest, 1)
        assert file_row(tmp_manifest, FILE_ID_A) is not None
        assert chunk_count(tmp_manifest, FILE_ID_A) == 1

    def test_rerun_is_idempotent_noop_on_target(self, engine, pg):
        self._seed_two_sources(engine)
        engine.delete_stale_sources(active_ids=[1])
        pg.reset()

        engine.delete_stale_sources(active_ids=[1])
        assert pg.deleted_ids() == []

    def test_no_stale_sources_sends_empty_deletes(self, engine, pg):
        engine.reconcile([
            make_file(FILE_PATH_A, [
                make_chunk(FILE_PATH_A, "a1", "active", ordinal=1),
            ], data_source_id=1),
        ])
        pg.reset()

        engine.delete_stale_sources(active_ids=[1])
        assert pg.deleted_ids() == []
