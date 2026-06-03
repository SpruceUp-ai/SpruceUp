import asyncio
import time
from dataclasses import dataclass, field

from ..base import EmbedderConnector


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
        )
        self._inner = inner
        self._max_wait = max_wait_ms / 1000
        self._max_batch_size = max_batch_size or getattr(inner, "max_batch_size", 50)
        self._semaphore = asyncio.Semaphore(max_concurrent_batches)
        self._all_pending_files: list[_PendingChunksForFile] = []
        self._wake = asyncio.Event()
        self._flusher_task: asyncio.Task | None = None

    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        return await self._inner.embed_batch(batch)

    async def process_chunks(self, chunks: list[str]) -> list[list[float]]:
        """Called by user's pipeline file, receives all the chunks from one file
        as input. Makes sure the batch-flusher exists, creates a future, then
        adds this file's chunks to the list of _pending chunks. Then wakes up
        the flusher to check whether it's time to create and dispatch a batch
        of chunks to the embedding model. Returns the future, which will be
        resolved when all of the embeddings for this file have been created."""
        if not chunks:
            return []
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
        return num_pending_chunks >= self._max_batch_size

    def _dispatch_pending_chunks(self) -> None:
        if not self._all_pending_files:
            return
        pending_files, self._all_pending_files = self._all_pending_files, []
        chunk_text_indexed_by_file_and_chunk = [
            (file_index, chunk_index, chunk_text)
            for file_index, pending_file in enumerate(pending_files)
            for chunk_index, chunk_text in enumerate(pending_file.chunks)
        ]
        for i in range(0, len(chunk_text_indexed_by_file_and_chunk), self._max_batch_size):
            batch = chunk_text_indexed_by_file_and_chunk[i : i + self._max_batch_size]
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
            except Exception as err:
                from ..base import EmbeddingError
                wrapped = EmbeddingError(str(err)) if not isinstance(err, EmbeddingError) else err
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
