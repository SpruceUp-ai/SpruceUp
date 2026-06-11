import asyncio
import time
from dataclasses import dataclass
from typing import cast

from ..base import EmbedderConnector
from ...utils.hashing import hash_text
from ...transform_context import get_transform_context


@dataclass
class _PendingChunk:
    text: str
    future: asyncio.Future


class EmbeddingBatcher(EmbedderConnector):
    def __init__(
        self,
        embedder: EmbedderConnector,
        max_wait_ms: int = 100,
        max_concurrent_batches: int = 5,
        max_batch_size: int | None = None,
    ) -> None:
        super().__init__(
            model=embedder.model,
            api_key=embedder.api_key,
            embedding_dimensions=embedder.embedding_dimensions,
            max_batch_size=max_batch_size or embedder.max_batch_size,
        )
        self._embedder = embedder
        self._max_wait = max_wait_ms / 1000
        self._semaphore = asyncio.Semaphore(max_concurrent_batches)
        self._pending: list[_PendingChunk] = []
        self._wake = asyncio.Event()
        self._batch_task: asyncio.Task | None = None

    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        return await self._embedder.embed_batch_retrying(batch)

    async def process_chunks(self, chunks: list[str]) -> list[list[float]]:
        if not chunks:
            return []

        ctx = get_transform_context()
        if ctx is None:
            return await self._dispatch_to_batcher(chunks)

        manifest = ctx.manifest
        file_id = ctx.file_id

        chunk_hashes = [hash_text(c) for c in chunks]
        cached = manifest.get_cached_embeddings(file_id, chunk_hashes)

        ctx.used_chunk_embedding_cache_keys.update(chunk_hashes)

        hits = {i: cached[h] for i, h in enumerate(chunk_hashes) if h in cached}
        ctx.embed_hits += len(hits)
        ctx.embed_total += len(chunks)

        miss_indices = [i for i, h in enumerate(chunk_hashes) if h not in cached]
        if not miss_indices:
            return [hits[i] for i in range(len(chunks))]

        miss_chunks = [chunks[i] for i in miss_indices]
        miss_hashes = [chunk_hashes[i] for i in miss_indices]
        miss_embeddings = await self._dispatch_to_batcher(miss_chunks)

        manifest.set_cached_embeddings(file_id, list(zip(miss_hashes, miss_embeddings)))

        results: list[list[float] | None] = [None] * len(chunks)
        for i, emb in hits.items():
            results[i] = emb
        for idx, emb in zip(miss_indices, miss_embeddings):
            results[idx] = emb
        return cast(list[list[float]], results)

    async def _dispatch_to_batcher(self, chunks: list[str]) -> list[list[float]]:
        self._ensure_batch_loop()
        loop = asyncio.get_running_loop()
        pending = [_PendingChunk(text=c, future=loop.create_future()) for c in chunks]
        self._pending.extend(pending)
        self._wake.set()
        return await asyncio.gather(*(pc.future for pc in pending))

    def _ensure_batch_loop(self) -> None:
        if self._batch_task is None or self._batch_task.done():
            self._batch_task = asyncio.create_task(self._batch_loop())

    async def _batch_loop(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            deadline = time.monotonic() + self._max_wait
            while True:
                if self._should_flush_now():
                    break
                remaining_time = deadline - time.monotonic()
                if remaining_time <= 0:
                    break
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=remaining_time)
                except asyncio.TimeoutError:
                    break
            self._dispatch_pending_chunks()

    def _should_flush_now(self) -> bool:
        return len(self._pending) >= self.max_batch_size

    def _dispatch_pending_chunks(self) -> None:
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        for i in range(0, len(pending), self.max_batch_size):
            batch = pending[i : i + self.max_batch_size]
            asyncio.create_task(self._run_batch(batch))

    async def aclose(self) -> None:
        if self._batch_task is not None and not self._batch_task.done():
            self._batch_task.cancel()
            await asyncio.gather(self._batch_task, return_exceptions=True)
        await self._embedder.aclose()

    async def _run_batch(self, batch: list[_PendingChunk]) -> None:
        from ..base import EmbeddingError

        texts = [pc.text for pc in batch]
        async with self._semaphore:
            try:
                embeddings = await self._embedder.embed_batch_retrying(texts)
                if len(embeddings) != len(texts):
                    raise EmbeddingError(
                        f"embedder returned {len(embeddings)} vector(s) for "
                        f"{len(texts)} input(s); counts must match"
                    )
            except Exception as err:
                wrapped = err if isinstance(err, EmbeddingError) else EmbeddingError(str(err))
                if wrapped is not err:
                    wrapped.__cause__ = err
                for pc in batch:
                    if not pc.future.done():
                        pc.future.set_exception(wrapped)
                return

        for pc, emb in zip(batch, embeddings):
            if not pc.future.done():
                pc.future.set_result(emb)
