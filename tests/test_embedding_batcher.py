import asyncio
import time

import pytest

from spruceup.connectors.base import EmbedderConnector
from spruceup.connectors.embedders.embedding_batcher import EmbeddingBatcher


def _fake_emb(s: str) -> list[float]:
    return [float(ord(c)) for c in s]


class FakeEmbedder(EmbedderConnector):
    def __init__(self, max_batch_size: int = 50, raise_exc: Exception | None = None):
        self.max_batch_size = max_batch_size
        self.calls: list[list[str]] = []
        self._raise_exc = raise_exc

    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        self.calls.append(list(batch))
        if self._raise_exc is not None:
            raise self._raise_exc
        return [_fake_emb(s) for s in batch]


@pytest.mark.asyncio
async def test_cross_caller_batching_merges_three_callers_into_one_batch():
    inner = FakeEmbedder(max_batch_size=50)
    embedder = EmbeddingBatcher(inner, max_wait_ms=50)

    results = await asyncio.gather(
        embedder.process_chunks(["a", "b"]),
        embedder.process_chunks(["c"]),
        embedder.process_chunks(["d", "e", "f"]),
    )

    assert len(inner.calls) == 1
    assert sorted(inner.calls[0]) == ["a", "b", "c", "d", "e", "f"]
    assert results[0] == [_fake_emb("a"), _fake_emb("b")]
    assert results[1] == [_fake_emb("c")]
    assert results[2] == [_fake_emb("d"), _fake_emb("e"), _fake_emb("f")]


@pytest.mark.asyncio
async def test_size_threshold_triggers_immediate_flush():
    inner = FakeEmbedder(max_batch_size=5)
    embedder = EmbeddingBatcher(inner, max_wait_ms=10_000)

    start = time.monotonic()
    result = await embedder.process_chunks(["a", "b", "c", "d", "e"])
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, "size threshold should bypass the time window"
    assert len(inner.calls) == 1
    assert inner.calls[0] == ["a", "b", "c", "d", "e"]
    assert result == [_fake_emb(s) for s in ["a", "b", "c", "d", "e"]]


@pytest.mark.asyncio
async def test_time_window_safety_net_fires():
    inner = FakeEmbedder(max_batch_size=50)
    embedder = EmbeddingBatcher(inner, max_wait_ms=50)

    start = time.monotonic()
    result = await embedder.process_chunks(["only"])
    elapsed = time.monotonic() - start

    assert elapsed >= 0.04, "should have waited roughly the time-window"
    assert len(inner.calls) == 1
    assert result == [_fake_emb("only")]


@pytest.mark.asyncio
async def test_per_caller_results_preserve_order_across_batches():
    # max_batch_size=3 forces 7 chunks into 3 batches; verify each caller's
    # results come back in the order they submitted them.
    inner = FakeEmbedder(max_batch_size=3)
    embedder = EmbeddingBatcher(inner, max_wait_ms=50)

    chunks = ["c1", "c2", "c3", "c4", "c5", "c6", "c7"]
    result = await embedder.process_chunks(chunks)

    assert len(inner.calls) >= 2
    assert result == [_fake_emb(s) for s in chunks]


@pytest.mark.asyncio
async def test_failure_in_inner_propagates_to_every_caller_in_batch():
    inner = FakeEmbedder(max_batch_size=50, raise_exc=RuntimeError("boom"))
    embedder = EmbeddingBatcher(inner, max_wait_ms=50)

    results = await asyncio.gather(
        embedder.process_chunks(["a"]),
        embedder.process_chunks(["b"]),
        return_exceptions=True,
    )

    assert all(isinstance(r, RuntimeError) for r in results)
    assert all("boom" in str(r) for r in results)


@pytest.mark.asyncio
async def test_empty_chunks_returns_immediately_without_dispatch():
    inner = FakeEmbedder()
    embedder = EmbeddingBatcher(inner, max_wait_ms=10_000)

    result = await embedder.process_chunks([])

    assert result == []
    assert inner.calls == []
