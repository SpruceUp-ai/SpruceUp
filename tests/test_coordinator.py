import asyncio
from dataclasses import dataclass

import pytest

from spruceup.connectors.base import EmbedderConnector, TargetConnector
from spruceup.coordinator import Coordinator
from spruceup.manifest import Manifest
from spruceup.memoize import memoize
from spruceup.memoize.context import (
    _memo_file_id_var,
    _memo_manifest_var,
    _memo_stats_var,
    _memo_temp_keys_var,
)
from spruceup.models import ChunkWrapper, SpruceFile, SyncTask
from spruceup.sync_engine import SyncEngine
from spruceup.utils.hashing import hash_source_ref


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

    @property
    def display_name(self) -> str:
        return "mock_target"

    def ensure_table_exists(self, embedding_dimensions: int) -> None:
        pass

    async def sync(self, upserts: list[ChunkWrapper], deletes: list) -> None:
        self.calls.append({"upserts": list(upserts), "deletes": list(deletes)})

    def inserted_ids(self) -> list:
        return [c.user_chunk.id for call in self.calls for c in call["upserts"]]

    def deleted_ids(self) -> list:
        return [pk for call in self.calls for pk in call["deletes"]]


class FakeEmbedder(EmbedderConnector):
    def __init__(self):
        super().__init__(embedding_dimensions=3)

    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        return [[float(len(s)), 0.0, 0.0] for s in batch]


class FakeSource:
    """Minimal SourceConnector surface the Coordinator actually calls."""

    def __init__(self, data_source_id: int, *, fetch_error: Exception | None = None):
        self.data_source_id = data_source_id
        self._fetch_error = fetch_error
        self.decode_calls: list[bytes] = []

    def display_name(self, identifier: str) -> str:
        return identifier.rsplit("/", 1)[-1]

    def decode_content(self, raw_content: bytes) -> str:
        self.decode_calls.append(raw_content)
        return raw_content.decode()

    async def fetch(self, task: SyncTask) -> SpruceFile:
        if self._fetch_error is not None:
            raise self._fetch_error
        ref = task.identifier
        # One line of content == one chunk.
        body = "alpha\nbeta\ngamma".encode()
        return SpruceFile(
            file_id=hash_source_ref(ref),
            source_ref=ref,
            display_name=self.display_name(ref),
            content_hash=hash_source_ref(body.decode()),
            file_type="txt",
            data_source_id=self.data_source_id,
            raw_content=body,
            chunks=[],
            source_metadata={"modified_at": 123.0},
        )


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

async def line_transform(*, file_props, embed) -> list[SimpleChunk]:
    lines = file_props.raw_content.splitlines()
    embeddings = await embed(lines)
    return [
        SimpleChunk(id=f"{file_props.source_ref}#{i}", chunk_text=line, chunk_embedding=emb)
        for i, (line, emb) in enumerate(zip(lines, embeddings))
    ]


async def bad_transform(*, file_props, embed):
    return "not a list"


@memoize(returns=list)
def _split_lines(text: str) -> list[str]:
    return text.splitlines()


async def memoizing_transform(*, file_props, embed) -> list[SimpleChunk]:
    lines = _split_lines(file_props.raw_content)
    embeddings = await embed(lines)
    return [
        SimpleChunk(id=f"{file_props.source_ref}#{i}", chunk_text=line, chunk_embedding=emb)
        for i, (line, emb) in enumerate(zip(lines, embeddings))
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PATH_A = "/test-corpus/doc_a.txt"
PATH_B = "/test-corpus/doc_b.txt"


@pytest.fixture
def target():
    return MockSyncTarget()


@pytest.fixture
def manifest(tmp_path):
    return Manifest(str(tmp_path / "manifest.db"))


@pytest.fixture
def source(manifest):
    ds_id = manifest.register_source("local", "/test-corpus")
    return FakeSource(ds_id)


def make_coordinator(manifest, target, source, transform=line_transform):
    engine = SyncEngine(manifest=manifest, target=target)
    return Coordinator(
        queue=asyncio.Queue(),
        transform=transform,
        embedder=FakeEmbedder(),
        sync_engine=engine,
        source_registry={source.data_source_id: source},
    )


def upsert_task(source, identifier=PATH_A):
    return SyncTask(
        source_type="local",
        identifier=identifier,
        change_type="upsert",
        data_source_id=source.data_source_id,
    )


# ---------------------------------------------------------------------------
# upsert path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_reconciles_and_writes_all_chunks_to_target(manifest, target, source):
    coord = make_coordinator(manifest, target, source)

    await coord.process_task(upsert_task(source))

    assert coord._failed_files == []
    assert target.inserted_ids() == [f"{PATH_A}#0", f"{PATH_A}#1", f"{PATH_A}#2"]
    assert target.deleted_ids() == []


@pytest.mark.asyncio
async def test_upsert_calls_source_decode_content(manifest, target, source):
    coord = make_coordinator(manifest, target, source)

    await coord.process_task(upsert_task(source))

    assert source.decode_calls == [b"alpha\nbeta\ngamma"]


@pytest.mark.asyncio
async def test_upsert_builds_chunk_wrappers_with_id_and_content_hash(manifest, target, source):
    coord = make_coordinator(manifest, target, source)

    await coord.process_task(upsert_task(source))

    wrappers = [c for call in target.calls for c in call["upserts"]]
    assert [w.ordinal for w in wrappers] == [0, 1, 2]
    assert all(isinstance(w.chunk_id, bytes) and len(w.chunk_id) == 16 for w in wrappers)
    assert all(isinstance(w.user_chunk_object_hash, bytes) for w in wrappers)


@pytest.mark.asyncio
async def test_reupsert_unchanged_file_writes_nothing_the_second_time(manifest, target, source):
    coord = make_coordinator(manifest, target, source)

    await coord.process_task(upsert_task(source))
    first_call_count = len(target.calls)
    await coord.process_task(upsert_task(source))

    assert len(target.calls) == first_call_count + 1
    assert target.calls[-1] == {"upserts": [], "deletes": []}


# ---------------------------------------------------------------------------
# memoize context wiring
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_sets_memoize_context_vars_for_the_transform(manifest, target, source):
    seen = {}

    async def capturing_transform(*, file_props, embed):
        seen["manifest"] = _memo_manifest_var.get()
        seen["file_id"] = _memo_file_id_var.get()
        seen["temp_keys"] = _memo_temp_keys_var.get()
        seen["stats"] = _memo_stats_var.get()
        return []

    coord = make_coordinator(manifest, target, source, transform=capturing_transform)
    await coord.process_task(upsert_task(source))

    assert seen["manifest"] is manifest
    assert seen["file_id"] == hash_source_ref(PATH_A)
    assert seen["temp_keys"] == set()
    assert seen["stats"] == [0, 0]


@pytest.mark.asyncio
async def test_upsert_logs_memoize_hits_on_second_run(manifest, target, source, caplog):
    coord = make_coordinator(manifest, target, source, transform=memoizing_transform)

    await coord.process_task(upsert_task(source))
    with caplog.at_level("INFO", logger="spruceup.coordinator"):
        await coord.process_task(upsert_task(source))

    assert coord._failed_files == []
    assert any("[memoize]" in r.message and "1/1 hits" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# delete + move dispatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_removes_previously_synced_chunks_from_target(manifest, target, source):
    coord = make_coordinator(manifest, target, source)
    await coord.process_task(upsert_task(source))

    delete = SyncTask(
        source_type="local",
        identifier=PATH_A,
        change_type="delete",
        data_source_id=source.data_source_id,
    )
    await coord.process_task(delete)

    assert coord._failed_files == []
    assert sorted(target.deleted_ids()) == [f"{PATH_A}#0", f"{PATH_A}#1", f"{PATH_A}#2"]


@pytest.mark.asyncio
async def test_move_does_not_touch_target(manifest, target, source):
    coord = make_coordinator(manifest, target, source)
    await coord.process_task(upsert_task(source))
    calls_after_upsert = len(target.calls)

    move = SyncTask(
        source_type="local",
        identifier=PATH_B,
        change_type="move",
        old_identifier=PATH_A,
        data_source_id=source.data_source_id,
    )
    await coord.process_task(move)

    assert coord._failed_files == []
    assert len(target.calls) == calls_after_upsert


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_error_is_swallowed_and_file_recorded_as_failed(manifest, target):
    ds_id = manifest.register_source("local", "/test-corpus")
    failing_source = FakeSource(ds_id, fetch_error=RuntimeError("network down"))
    coord = make_coordinator(manifest, target, failing_source)

    await coord.process_task(upsert_task(failing_source))

    assert coord._failed_files == ["doc_a.txt"]
    assert target.calls == []


@pytest.mark.asyncio
async def test_invalid_transform_output_is_recorded_as_failed(manifest, target, source):
    coord = make_coordinator(manifest, target, source, transform=bad_transform)

    await coord.process_task(upsert_task(source))

    assert coord._failed_files == ["doc_a.txt"]
    assert target.calls == []


# ---------------------------------------------------------------------------
# run() loop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_drains_queue_and_marks_tasks_done(manifest, target, source):
    coord = make_coordinator(manifest, target, source)
    await coord._queue.put(upsert_task(source, PATH_A))
    await coord._queue.put(upsert_task(source, PATH_B))

    runner = asyncio.create_task(coord.run())
    await asyncio.wait_for(coord._queue.join(), timeout=2.0)
    runner.cancel()
    try:
        await runner
    except asyncio.CancelledError:
        pass

    assert coord._failed_files == []
    assert sorted(target.inserted_ids()) == [
        f"{PATH_A}#0", f"{PATH_A}#1", f"{PATH_A}#2",
        f"{PATH_B}#0", f"{PATH_B}#1", f"{PATH_B}#2",
    ]
