"""Stub connectors for load testing SpruceUp's pipeline in isolation.

The stub embedder removes the real embedding API (cost + rate limits + network
latency) so the pipeline's own mechanics become the bottleneck under test.
The stub target removes Postgres so the pipeline can be measured without the
DB as a variable. Both can simulate latency to mimic the real dependency.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from spruceup.connectors.base import EmbedderConnector, TargetConnector
from spruceup.models import ChunkWrapper


@dataclass
class LoadTestChunk:
    id: str
    chunk_text: str
    chunk_embedding: list[float]
    source_file: str


class StubEmbedder(EmbedderConnector):
    """Returns fixed-dimension zero vectors. Optional per-batch latency to
    mimic the round-trip of a real embedding API."""

    def __init__(
        self,
        dimensions: int = 1536,
        latency_s: float = 0.0,
        max_batch_size: int = 150,
    ) -> None:
        super().__init__(embedding_dimensions=dimensions)
        self.max_batch_size = max_batch_size
        self._latency_s = latency_s
        self.batch_calls = 0
        self.chunks_embedded = 0

    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        self.batch_calls += 1
        self.chunks_embedded += len(batch)
        if self._latency_s:
            await asyncio.sleep(self._latency_s)
        vec = [0.0] * self.embedding_dimensions
        return [list(vec) for _ in batch]


@dataclass
class StubTarget(TargetConnector):
    """No-op target that counts what it was asked to write."""

    schema: type = LoadTestChunk
    primary_key: str = "id"
    latency_s: float = 0.0
    upserts: int = field(default=0, init=False)
    deletes: int = field(default=0, init=False)
    sync_calls: int = field(default=0, init=False)

    @property
    def display_name(self) -> str:
        return "stub"

    def ensure_table_exists(self, embedding_dimensions: int) -> None:
        pass

    def sync(self, upserts: list[ChunkWrapper], deletes: list) -> None:
        self.sync_calls += 1
        self.upserts += len(upserts)
        self.deletes += len(deletes)
        if self.latency_s:
            import time
            time.sleep(self.latency_s)
