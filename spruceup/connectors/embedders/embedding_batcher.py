import asyncio
import time
from dataclasses import dataclass, field

from ..base import EmbedderConnector
from ...utils.hashing import hash_text
from ...transform_context import _transform_context


@dataclass
class _PendingChunksForFile:
    chunks: list[str]
    future: asyncio.Future
    results: list[list[float] | None] = field(default_factory=list)


class EmbeddingBatcher(EmbedderConnector):
    def __init__(
        self,
        inner: EmbedderConnector,
        max_wait_ms: int = 100,
        max_concurrent_batches: int = 5,
        max_batch_size: int | None = None,
    ) -> None:
        super().__init__(
            model=inner.model,
            api_key=inner.api_key,
            embedding_dimensions=inner.embedding_dimensions,
            max_batch_size=max_batch_size or inner.max_batch_size,
        )
        self._inner = inner
        self._max_wait = max_wait_ms / 1000
        self._semaphore = asyncio.Semaphore(max_concurrent_batches)
        self._all_pending_files: list[_PendingChunksForFile] = []
        self._wake = asyncio.Event()
        self._flusher_task: asyncio.Task | None = None

    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        return await self._inner.embed_batch(batch)

    async def process_chunks(self, chunks: list[str]) -> list[list[float]]:
        if not chunks:
            return []

        ctx = _transform_context.get()
        if ctx is None:
            return await self._dispatch_to_batcher(chunks)

        manifest = ctx.manifest
        file_id = ctx.file_id

        chunk_hashes = [hash_text(c) for c in chunks]
        cached = manifest.get_cached_embeddings(file_id, chunk_hashes)

        ctx.embed_used_hashes.update(chunk_hashes)

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

        results = [None] * len(chunks)
        for i, emb in hits.items():
            results[i] = emb
        for idx, emb in zip(miss_indices, miss_embeddings):
            results[idx] = emb
        return results

    async def _dispatch_to_batcher(self, chunks: list[str]) -> list[list[float]]:
        self._ensure_flusher()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._all_pending_files.append(
            _PendingChunksForFile(
                chunks=list(chunks),
                future=future,
                results=[None] * len(chunks),
            )
        )
        self._wake.set()
        return await future

    def _ensure_flusher(self) -> None:
        if self._flusher_task is None or self._flusher_task.done():
            self._flusher_task = asyncio.create_task(self._flusher_loop())

    async def _flusher_loop(self) -> None:
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
        if not self._all_pending_files:
            return False
        num_pending_chunks = sum(len(pending_file.chunks) for pending_file in self._all_pending_files)
        return num_pending_chunks >= self.max_batch_size

    def _dispatch_pending_chunks(self) -> None:
        if not self._all_pending_files:
            return
        pending_files, self._all_pending_files = self._all_pending_files, []
        chunk_text_indexed_by_file_and_chunk = [
            (file_index, chunk_index, chunk_text)
            for file_index, pending_file in enumerate(pending_files)
            for chunk_index, chunk_text in enumerate(pending_file.chunks)
        ]
        for i in range(0, len(chunk_text_indexed_by_file_and_chunk), self.max_batch_size):
            batch = chunk_text_indexed_by_file_and_chunk[i : i + self.max_batch_size]
            asyncio.create_task(self._run_batch(batch, pending_files))

    async def aclose(self) -> None:
        if self._flusher_task is not None and not self._flusher_task.done():
            self._flusher_task.cancel()
            await asyncio.gather(self._flusher_task, return_exceptions=True)
        await self._inner.aclose()

    async def _run_batch(
        self,
        batch: list[tuple[int, int, str]],
        files_in_flight: list[_PendingChunksForFile],
    ) -> None:
        batch_strs = [chunk_text for _, _, chunk_text in batch]
        touched_files = {file_index for file_index, _, _ in batch}
        async with self._semaphore:
            try:
                embeddings = await self._inner.embed_batch(batch_strs)
                if len(embeddings) != len(batch_strs):
                    from ..base import EmbeddingError
                    raise EmbeddingError(
                        f"embedder returned {len(embeddings)} vector(s) for "
                        f"{len(batch_strs)} input(s); counts must match"
                    )
            except Exception as err:
                from ..base import EmbeddingError
                if isinstance(err, EmbeddingError):
                    wrapped = err
                else:
                    wrapped = EmbeddingError(str(err))
                    wrapped.__cause__ = err
                for file_index in touched_files:
                    if not files_in_flight[file_index].future.done():
                        files_in_flight[file_index].future.set_exception(wrapped)
                return

        for (file_index, chunk_index, _), embedding in zip(batch, embeddings):
            files_in_flight[file_index].results[chunk_index] = embedding

        for file_index in touched_files:
            file = files_in_flight[file_index]
            if not file.future.done() and all(result is not None for result in file.results):
                file.future.set_result(file.results)
