import asyncio
import os

import openai
import tenacity

from ..base import EmbedderConnector


class OpenAIEmbedder(EmbedderConnector):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        max_batch_size: int = 50,
        max_concurrent_batches: int = 5,
    ) -> None:
        self._model = model
        self._max_batch_size = max_batch_size
        self._max_concurrent_batches = max_concurrent_batches
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self._client = openai.AsyncOpenAI(api_key=api_key)
        return self._client

    @tenacity.retry(
        wait=tenacity.wait_exponential_jitter(initial=1, max=30),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        response = await self._get_client().embeddings.create(
            model=self._model,
            input=batch,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]

    async def process_chunks(self, chunks: list[str]) -> list[list[float]]:
        if not chunks:
            return []
        semaphore = asyncio.Semaphore(self._max_concurrent_batches)
        batches = [
            chunks[i:i + self._max_batch_size]
            for i in range(0, len(chunks), self._max_batch_size)
        ]

        async def _embed_with_limit(batch: list[str]) -> list[list[float]]:
            async with semaphore:
                return await self._embed_batch(batch)

        results = await asyncio.gather(*(_embed_with_limit(b) for b in batches))
        return [embedding for batch_result in results for embedding in batch_result]
