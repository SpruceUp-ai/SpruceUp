import asyncio
import sqlite3
import types
from dataclasses import dataclass

import pytest

import spruceup.app as app
from spruceup.connectors.base import EmbedderConnector, TargetConnector
from spruceup.manifest import Manifest
from spruceup.models import ChunkWrapper, SpruceFile, SyncTask
from spruceup.monitoring.monitor import BaseWatcher
from spruceup.utils.hashing import hash_source_ref, hash_transform


# ---------------------------------------------------------------------------
# Test schema + fakes
# ---------------------------------------------------------------------------

@dataclass
class SimpleChunk:
    id: str
    chunk_text: str
    chunk_embedding: list[float]


class MockSyncTarget(TargetConnector):
    primary_key = "id"
    schema = SimpleChunk

    def __init__(self):
        self.calls: list[dict] = []
        self.ensure_dims: int | None = None
        self.closed = False

    @property
    def display_name(self) -> str:
        return "mock_target"

    def ensure_table_exists(self, embedding_dimensions: int) -> None:
        self.ensure_dims = embedding_dimensions

    async def sync(self, upserts: list[ChunkWrapper], deletes: list) -> None:
        self.calls.append({"upserts": list(upserts), "deletes": list(deletes)})

    def close(self) -> None:
        self.closed = True

    def inserted_ids(self) -> list:
        return [c.user_chunk.id for call in self.calls for c in call["upserts"]]


class FakeEmbedder(EmbedderConnector):
    def __init__(self):
        super().__init__(embedding_dimensions=3)

    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        return [[float(len(s)), 0.0, 0.0] for s in batch]


class FakeWatcher(BaseWatcher):
    """Enqueues a fixed set of upsert tasks during catch-up, then idles."""

    def __init__(self, data_source_id: int, identifiers: list[str], seen: dict):
        self._ds_id = data_source_id
        self._identifiers = identifiers
        self._seen = seen

    @property
    def source_type(self) -> str:
        return "local"

    async def _catch_up(self, queue, manifest, force_reindex=False):
        self._seen["force_reindex"] = force_reindex
        for ident in self._identifiers:
            await queue.put(
                SyncTask(
                    source_type="local",
                    identifier=ident,
                    change_type="upsert",
                    data_source_id=self._ds_id,
                )
            )

    async def _watch(self, queue, manifest, catchup_done):
        await asyncio.Event().wait()  # idle until cancelled


class FakeSource:
    """SourceConnector surface that app + coordinator actually touch."""

    validate_calls: list = []

    def __init__(self, identifiers, *, transform_blowup_on=None):
        self._identifiers = identifiers
        self._seen: dict = {}

    @property
    def source_type(self) -> str:
        return "local"

    @property
    def source_identifier(self) -> str:
        return "/test-corpus"

    def create_watcher(self, data_source_id: int) -> FakeWatcher:
        return FakeWatcher(data_source_id, self._identifiers, self._seen)

    @classmethod
    async def validate(cls, sources) -> None:
        cls.validate_calls.append(list(sources))

    def display_name(self, identifier: str) -> str:
        return identifier.rsplit("/", 1)[-1]

    def decode_content(self, raw_content: bytes) -> str:
        return raw_content.decode()

    async def fetch(self, task: SyncTask) -> SpruceFile:
        ref = task.identifier
        body = b"alpha\nbeta"
        return SpruceFile(
            file_id=hash_source_ref(ref),
            source_ref=ref,
            display_name=self.display_name(ref),
            content_hash=hash_source_ref(body.decode()),
            file_type="txt",
            data_source_id=task.data_source_id,
            raw_content=body,
            chunks=[],
            source_metadata={"modified_at": 1.0},
        )

    def __repr__(self) -> str:
        return "FakeSource(/test-corpus)"


# ---------------------------------------------------------------------------
# Transforms (module-level so hash_transform can read their source)
# ---------------------------------------------------------------------------

async def good_transform(*, file_props, embed) -> list[SimpleChunk]:
    lines = file_props.raw_content.splitlines()
    embeddings = await embed(lines)
    return [
        SimpleChunk(id=f"{file_props.source_ref}#{i}", chunk_text=line, chunk_embedding=emb)
        for i, (line, emb) in enumerate(zip(lines, embeddings))
    ]


async def failing_transform(*, file_props, embed) -> list[SimpleChunk]:
    raise RuntimeError("transform blew up")


PATH_A = "/test-corpus/doc_a.txt"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_validate_calls():
    FakeSource.validate_calls = []
    yield


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    m = Manifest(str(tmp_path / "manifest.db"))
    monkeypatch.setattr(app, "Manifest", lambda: m)
    return m


def make_pipeline(target, transform, identifiers=(PATH_A,)):
    config = types.SimpleNamespace(
        target=target,
        transform=transform,
        embedder=FakeEmbedder(),
        sources=[FakeSource(list(identifiers))],
    )
    return types.SimpleNamespace(config=config)


def reopen(manifest):
    return Manifest(manifest._path)


async def _wait_until(pred, timeout=3.0):
    async def loop():
        while not pred():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(loop(), timeout)


async def run_app_until(pipeline, pred, timeout=3.0):
    task = asyncio.create_task(app.run(pipeline))
    try:
        await _wait_until(pred, timeout)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    return task


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wires_components_and_processes_catch_up_tasks(manifest):
    target = MockSyncTarget()
    pipeline = make_pipeline(target, good_transform)

    await run_app_until(pipeline, lambda: target.inserted_ids())

    assert target.ensure_dims == 3
    assert len(FakeSource.validate_calls) == 1
    assert target.inserted_ids() == [f"{PATH_A}#0", f"{PATH_A}#1"]


@pytest.mark.asyncio
async def test_force_reindex_records_transform_hash_after_clean_run(manifest):
    target = MockSyncTarget()
    pipeline = make_pipeline(target, good_transform)
    h = hash_transform(good_transform)

    assert manifest.transform_hash_changed(h) is True

    await run_app_until(pipeline, lambda: not manifest.transform_hash_changed(h))

    assert reopen(manifest).transform_hash_changed(h) is False
    assert pipeline.config.sources[0]._seen["force_reindex"] is True


@pytest.mark.asyncio
async def test_incremental_run_skips_reindex_when_hash_unchanged(manifest):
    target = MockSyncTarget()
    pipeline = make_pipeline(target, good_transform)
    h = hash_transform(good_transform)
    manifest.update_transform_hash(h)  # pre-record -> no reindex

    await run_app_until(pipeline, lambda: target.inserted_ids())

    assert reopen(manifest).transform_hash_changed(h) is False
    assert pipeline.config.sources[0]._seen["force_reindex"] is False
    assert target.inserted_ids() == [f"{PATH_A}#0", f"{PATH_A}#1"]


@pytest.mark.asyncio
async def test_failed_file_during_reindex_leaves_hash_unrecorded(manifest, caplog):
    target = MockSyncTarget()
    pipeline = make_pipeline(target, failing_transform)
    h = hash_transform(failing_transform)

    with caplog.at_level("WARNING", logger="spruceup.app"):
        await run_app_until(
            pipeline,
            lambda: any("Reindex incomplete" in r.message for r in caplog.records),
        )

    assert reopen(manifest).transform_hash_changed(h) is True
    assert target.inserted_ids() == []


@pytest.mark.asyncio
async def test_shutdown_closes_target_and_manifest_connection(manifest):
    target = MockSyncTarget()
    pipeline = make_pipeline(target, good_transform)

    await run_app_until(pipeline, lambda: target.inserted_ids())

    assert target.closed is True
    with pytest.raises(sqlite3.ProgrammingError):
        manifest._conn.execute("SELECT 1")
