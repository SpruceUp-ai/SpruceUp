import asyncio
import time
from dataclasses import dataclass, field

from ..base import EmbedderConnector


@dataclass
class _Pending:
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
        self._inner = inner
        self._max_wait = max_wait_ms / 1000
        self._max_batch_size = max_batch_size or getattr(inner, "max_batch_size", 50)
        self._semaphore = asyncio.Semaphore(max_concurrent_batches)
        self._pending: list[_Pending] = []
        self._expected = 0
        self._submissions_received = 0
        self._wake = asyncio.Event()
        self._flusher_task: asyncio.Task | None = None

    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        return await self._inner.embed_batch(batch)

    def expect(self, n: int = 1) -> None:
        self._expected += n

    async def process_chunks(self, chunks: list[str]) -> list[list[float]]:
        if not chunks:
            return []
        self._ensure_flusher()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending.append(
            _Pending(
                chunks=list(chunks),
                future=future,
                results=[None] * len(chunks),
            )
        )
        self._submissions_received += 1
        self._wake.set()
        return await future

    def _ensure_flusher(self) -> None:
        if self._flusher_task is None or self._flusher_task.done():
            self._flusher_task = asyncio.create_task(self._flusher_loop())

    async def _flusher_loop(self) -> None:
        while True:
            await self._wake.wait()
            deadline = time.monotonic() + self._max_wait
            while True:
                self._wake.clear()
                if self._should_flush_now():
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
            self._dispatch_pending()

    def _should_flush_now(self) -> bool:
        if not self._pending:
            return False
        pool_size = sum(len(p.chunks) for p in self._pending)
        if pool_size >= self._max_batch_size:
            return True
        if self._expected > 0 and self._submissions_received >= self._expected:
            return True
        return False

    def _dispatch_pending(self) -> None:
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        consumed = min(self._submissions_received, self._expected)
        self._expected -= consumed
        self._submissions_received -= consumed

        flat = [
            (pi, ci, s)
            for pi, p in enumerate(pending)
            for ci, s in enumerate(p.chunks)
        ]
        for i in range(0, len(flat), self._max_batch_size):
            slice_ = flat[i : i + self._max_batch_size]
            asyncio.create_task(self._run_batch(slice_, pending))

    async def _run_batch(
        self,
        slice_: list[tuple[int, int, str]],
        pending: list[_Pending],
    ) -> None:
        batch_strs = [s for _, _, s in slice_]
        async with self._semaphore:
            try:
                embeddings = await self._inner.embed_batch(batch_strs)
            except Exception as e:
                touched = {pi for pi, _, _ in slice_}
                for pi in touched:
                    if not pending[pi].future.done():
                        pending[pi].future.set_exception(e)
                return

        for (pi, ci, _), emb in zip(slice_, embeddings):
            pending[pi].results[ci] = emb

        touched = {pi for pi, _, _ in slice_}
        for pi in touched:
            p = pending[pi]
            if not p.future.done() and all(r is not None for r in p.results):
                p.future.set_result(p.results)
