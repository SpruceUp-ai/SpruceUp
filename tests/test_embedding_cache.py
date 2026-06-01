"""
Tests for the content-scoped embedding cache.

SQLite manifest runs against a real temp file. The "inner" embedder is faked so
we can count exactly which texts reach the API.
"""

from dataclasses import dataclass

import pytest

from spruceup.connectors.base import EmbedderConnector, TargetConnector
from spruceup.connectors.embedders.caching import CachingEmbedder
from spruceup.connectors.embedders.embedding_batcher import EmbeddingBatcher
from spruceup.coordinator import Coordinator
from spruceup.manifest import Manifest
from spruceup.memoize.context import _embed_text_hashes_var
from spruceup.models import SpruceFile, SyncTask
from spruceup.sync_engine import SyncEngine
from spruceup.utils.hashing import (
    hash_chunk_text,
    hash_source_ref,
    pack_floats,
    unpack_floats,
)

SPEC = "text-embedding-3-small:1536"
OTHER_SPEC = "text-embedding-3-large:3072"


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


def _fake_vec(s: str) -> list[float]:
    # Distinct, deterministic, float32-exact vector per string.
    return [float(ord(c)) for c in s] or [0.0]


class FakeInner(EmbedderConnector):
    """Records every text it is asked to embed, in order."""

    def __init__(self):
        super().__init__(embedding_dimensions=1536)
        self.seen: list[str] = []

    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        self.seen.extend(batch)
        return [_fake_vec(s) for s in batch]

    async def process_chunks(self, chunks: list[str]) -> list[list[float]]:
        return await self.embed_batch(chunks)


@pytest.fixture
def manifest(tmp_path):
    return Manifest(str(tmp_path / "manifest.db"))


@pytest.fixture
def inner():
    return FakeInner()


@pytest.fixture
def embedder(inner, manifest):
    return CachingEmbedder(inner, manifest=manifest, embedding_spec=SPEC)


# ---------------------------------------------------------------------------
# pack/unpack round-trip
# ---------------------------------------------------------------------------


def test_pack_unpack_float32_round_trip():
    v = [0.0, 1.5, -2.25, 3.140625, 1e10]  # all float32-exact
    assert unpack_floats(pack_floats(v)) == v


# ---------------------------------------------------------------------------
# Hit / miss split
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_miss_calls_inner_and_stores(embedder, inner, manifest):
    result = await embedder.process_chunks(["alpha", "beta"])

    assert inner.seen == ["alpha", "beta"]
    assert result == [_fake_vec("alpha"), _fake_vec("beta")]
    # Both vectors now cached under SPEC.
    cached = manifest.get_embeddings(
        [hash_chunk_text("alpha"), hash_chunk_text("beta")], SPEC
    )
    assert len(cached) == 2


@pytest.mark.asyncio
async def test_hit_does_not_call_inner(embedder, inner, manifest):
    await embedder.process_chunks(["alpha", "beta"])
    inner.seen.clear()

    result = await embedder.process_chunks(["alpha", "beta"])

    assert inner.seen == []  # pure cache hit
    assert result == [_fake_vec("alpha"), _fake_vec("beta")]


@pytest.mark.asyncio
async def test_inner_called_only_for_misses(embedder, inner, manifest):
    # Pre-seed "alpha" only.
    manifest.set_embeddings([(hash_chunk_text("alpha"), SPEC, _fake_vec("alpha"))])

    await embedder.process_chunks(["alpha", "beta"])

    assert inner.seen == ["beta"]  # alpha hit, beta missed


# ---------------------------------------------------------------------------
# Order preservation across an interleaved hit/miss batch (load-bearing)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_order_preserved_interleaved(embedder, inner, manifest):
    # Seed positions 0, 2, 3 as hits; 1, 4 will miss. Pattern [hit,miss,hit,hit,miss].
    texts = ["h0", "m1", "h2", "h3", "m4"]
    for t in ("h0", "h2", "h3"):
        manifest.set_embeddings([(hash_chunk_text(t), SPEC, _fake_vec(t))])

    result = await embedder.process_chunks(texts)

    assert inner.seen == ["m1", "m4"]  # only misses, in order
    assert result == [_fake_vec(t) for t in texts]  # full result in ORIGINAL order


# ---------------------------------------------------------------------------
# Spec scoping — same text under a different spec misses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_different_spec_misses(inner, manifest):
    # Cache "alpha" under SPEC, then look it up under OTHER_SPEC.
    manifest.set_embeddings([(hash_chunk_text("alpha"), SPEC, _fake_vec("alpha"))])
    other = CachingEmbedder(inner, manifest=manifest, embedding_spec=OTHER_SPEC)

    await other.process_chunks(["alpha"])

    assert inner.seen == ["alpha"]  # different space → miss


# ---------------------------------------------------------------------------
# Metadata-only edit (text unchanged) → hit, zero API calls (the headline case)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metadata_edit_text_unchanged_hits(embedder, inner):
    # First "run": text embedded.
    await embedder.process_chunks(["lecture body text"])
    inner.seen.clear()
    # A metadata-only transform edit reruns with the SAME embeddable text.
    await embedder.process_chunks(["lecture body text"])
    assert inner.seen == []


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_input_no_touch(embedder, inner, manifest):
    assert await embedder.process_chunks([]) == []
    assert inner.seen == []
    assert manifest.embedding_cache_size() == 0


# ---------------------------------------------------------------------------
# Provenance — ContextVar records exactly the hashes the cache used
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provenance_contextvar_records_text_hashes(embedder):
    bucket: list[bytes] = []
    _embed_text_hashes_var.set(bucket)

    await embedder.process_chunks(["one", "two", "three"])

    assert bucket == [
        hash_chunk_text("one"),
        hash_chunk_text("two"),
        hash_chunk_text("three"),
    ]


@pytest.mark.asyncio
async def test_provenance_no_contextvar_is_safe(inner, manifest):
    # No transform context active → must not raise.
    _embed_text_hashes_var.set(None)
    e = CachingEmbedder(inner, manifest=manifest, embedding_spec=SPEC)
    assert await e.process_chunks(["x"]) == [_fake_vec("x")]


# ---------------------------------------------------------------------------
# Wipe (spec change) — empties the cache up front
# ---------------------------------------------------------------------------


def test_wipe_empties_cache(manifest):
    manifest.set_embeddings([(hash_chunk_text("a"), SPEC, _fake_vec("a"))])
    assert manifest.embedding_cache_size() == 1

    manifest.wipe_embedding_cache()

    assert manifest.embedding_cache_size() == 0


def test_embedding_spec_changed_first_run_then_recorded(manifest):
    assert manifest.embedding_spec_changed(SPEC) is True  # nothing stored yet
    manifest.update_embedding_spec(SPEC)
    assert manifest.embedding_spec_changed(SPEC) is False
    assert manifest.embedding_spec_changed(OTHER_SPEC) is True


# ---------------------------------------------------------------------------
# Sweep — orphaned rows removed, live rows retained
# ---------------------------------------------------------------------------


def test_sweep_removes_orphans_keeps_live(manifest):
    live_hash = hash_chunk_text("live")
    orphan_hash = hash_chunk_text("orphan")
    manifest.set_embeddings([
        (live_hash, SPEC, _fake_vec("live")),
        (orphan_hash, SPEC, _fake_vec("orphan")),
    ])

    # A live chunk references only live_hash.
    conn = manifest.connect()
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO files (id, source_ref) VALUES (?, ?)",
            (b"\x01" * 16, "corpus/doc.txt"),
        )
        conn.execute(
            "INSERT INTO chunks (id, file_id, text_hash) VALUES (?, ?, ?)",
            (b"\x02" * 16, b"\x01" * 16, live_hash),
        )

    deleted = manifest.sweep_embedding_cache()

    assert deleted == 1
    assert manifest.get_embeddings([live_hash], SPEC)  # retained
    assert not manifest.get_embeddings([orphan_hash], SPEC)  # swept


# ---------------------------------------------------------------------------
# embedding_spec property — the base-class default's three branches
# ---------------------------------------------------------------------------


def test_embedding_spec_property_resolution():
    # Concrete embedder: builds "{_model}:{dimensions}" from its own fields.
    class FakeConcrete(EmbedderConnector):
        def __init__(self):
            super().__init__(embedding_dimensions=1536)
            self._model = "text-embedding-3-small"

        async def embed_batch(self, batch):
            return [[0.0] for _ in batch]

    concrete = FakeConcrete()
    assert concrete.embedding_spec == "text-embedding-3-small:1536"

    # Wrapper: delegates down its _embedder_connector chain. Both real wrappers
    # store that field, so this pins the base default ↔ field-name agreement.
    batcher = EmbeddingBatcher(concrete)
    assert batcher.embedding_spec == "text-embedding-3-small:1536"

    # Neither _embedder_connector nor _model → loud failure, not a silent ""...
    class FakeNeither(EmbedderConnector):
        async def embed_batch(self, batch):
            return []

    with pytest.raises(NotImplementedError):
        _ = FakeNeither().embedding_spec


# ---------------------------------------------------------------------------
# Coordinator-level provenance — chunks.text_hash equals the hash the cache used
# ---------------------------------------------------------------------------


@dataclass
class _ProvChunk:
    id: str
    chunk_text: str
    chunk_embedding: list[float]
    lecture_title: str  # metadata field — proves text_hash ignores it


class _FakeTarget(TargetConnector):
    """Records sync calls; carries the schema + primary_key the coordinator reads."""

    schema = _ProvChunk
    primary_key = "id"

    def __init__(self):
        self.calls: list[dict] = []

    @property
    def display_name(self) -> str:
        return "fake_target"

    def ensure_table_exists(self, embedding_dimensions: int) -> None:
        pass

    async def sync(self, upserts, deletes) -> None:
        self.calls.append({"upserts": list(upserts), "deletes": list(deletes)})


class _FakeSource:
    """Minimal SourceConnector surface the coordinator's upsert path touches."""

    def __init__(self, source_ref: str, chunk_texts: list[str]):
        self._source_ref = source_ref
        self._chunk_texts = chunk_texts

    def display_name(self, identifier: str) -> str:
        return identifier.rsplit("/", 1)[-1]

    def decode_content(self, raw_content) -> str:
        return ""  # the transform reads chunk_texts from the closure, not raw_content

    async def fetch(self, task: SyncTask) -> SpruceFile:
        fid = hash_source_ref(self._source_ref)
        return SpruceFile(
            file_id=fid,
            source_ref=self._source_ref,
            display_name=self.display_name(self._source_ref),
            content_hash=fid,
            file_type="txt",
            data_source_id=1,
            raw_content=b"",
            chunks=[],
            source_metadata={"modified_at": 1_000_000.0},
        )


def _persisted_text_hashes(manifest: Manifest, file_id: bytes) -> list[bytes]:
    # chunks has no ordinal column; rowid preserves insert order, which matches
    # the order upsert_chunks wrote them (= chunk order = embed order).
    return [
        row[0]
        for row in manifest.connect().execute(
            "SELECT text_hash FROM chunks WHERE file_id = ? ORDER BY rowid", (file_id,)
        )
    ]


def _make_coordinator(manifest: Manifest, source: _FakeSource, transform):
    target = _FakeTarget()
    sync_engine = SyncEngine(manifest=manifest, target=target)
    manifest.register_source("local", "/corpus")
    return Coordinator(
        queue=object(),
        transform=transform,
        embedder=CachingEmbedder(
            EmbeddingBatcher(FakeInner()), manifest=manifest, embedding_spec=SPEC
        ),
        sync_engine=sync_engine,
        source_registry={1: source},
    )


@pytest.mark.asyncio
async def test_coordinator_persists_text_hash_matching_cache(manifest):
    texts = ["chunk one", "chunk two", "chunk three"]
    source = _FakeSource("/corpus/lecture.txt", texts)

    async def transform(*, file_props, embed):
        embeddings = await embed(texts)
        return [
            _ProvChunk(id=f"id{i}", chunk_text=t, chunk_embedding=e, lecture_title="T")
            for i, (t, e) in enumerate(zip(texts, embeddings))
        ]

    coord = _make_coordinator(manifest, source, transform)
    task = SyncTask(source_type="local", identifier="/corpus/lecture.txt",
                    change_type="upsert", data_source_id=1)
    await coord.upsert_file(task, "lecture.txt", source)

    file_id = hash_source_ref("/corpus/lecture.txt")
    persisted = _persisted_text_hashes(manifest, file_id)
    assert persisted == [hash_chunk_text(t) for t in texts]


@pytest.mark.asyncio
async def test_coordinator_drops_provenance_on_count_mismatch(manifest):
    # Transform embeds 2 texts but returns 3 chunks → counts disagree → all None.
    embedded = ["a", "b"]
    source = _FakeSource("/corpus/odd.txt", embedded)

    async def transform(*, file_props, embed):
        await embed(embedded)  # 2 hashes recorded
        return [  # 3 chunks returned
            _ProvChunk(id=f"id{i}", chunk_text=t, chunk_embedding=[0.0], lecture_title="T")
            for i, t in enumerate(["a", "b", "c"])
        ]

    coord = _make_coordinator(manifest, source, transform)
    task = SyncTask(source_type="local", identifier="/corpus/odd.txt",
                    change_type="upsert", data_source_id=1)
    await coord.upsert_file(task, "odd.txt", source)

    file_id = hash_source_ref("/corpus/odd.txt")
    persisted = _persisted_text_hashes(manifest, file_id)
    assert persisted == [None, None, None]  # no mis-attribution
