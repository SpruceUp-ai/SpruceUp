import openai
import tenacity
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# Custom embedding provider classes to allow for future extensibility.
class EmbeddingProvider:
    """Abstract base class for embedding providers."""

    def __init__(self, provider: str, model: str, encoding_format: str = "float", dimensions: int = 512):
        self._provider = provider
        self._model = model
        self._encoding_format = encoding_format
        self._dimensions = dimensions

    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider implementation. Client is reused across calls."""

    def __init__(self, model: str = "text-embedding-3-small", encoding_format: str = "float"):
        super().__init__("openai", model, encoding_format)
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self._client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def embed_batch(self, batch: list[str]) -> list[list[float]]:

        response = await self._client.embeddings.create(
            model=self._model,
            input=batch,
            dimensions=self._dimensions
        )

        return [item.embedding for item in response.data]


# --------------------------------------------

# `chunks`: ["abc", "def", "ghi", "123", "456"]
#
# `batch_chunks()` -> `batches`: [["abc", "def"], ["ghi", "123"], ["456"]]
#   len(chunks): 5
#   _max_batch_size: 2
#   chunks[0...2] -> ["abc", "def"]
#   chunks[2...4] -> ["ghi", "123"]
#   chunks[4...5] -> ["456"]
#
# for batch in batches:
#   `@tenacity.retry(...) def embed_batch(batch)` -> `batch_embeddings: list[list[float]]`
#
# -> `embeddings`: [ [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15 ] ]

class Embedder:
    """
    A long-lived service-object that generates vector embeddings for text chunks.
    Chunks are processed in batches.
    Uses back-off retry on failure.

    Attributes:
    -----------
    provider: Object?
        The provider to use for embedding.
    max_batch_size: int
        Max number of chunks to embed in a single batch.
    max_concurrent_batches: int
        Max number of concurrent batches to process.

    Behavior:
    ---------
    batch_chunks(chunks) -> list[list[str]]
        Generate list of `batches` by splitting `chunks` by `max_batch_size`.
    embed_batch: async
        Embed `chunks` in batches.
        Decorated with `tenacity.retry` to handle transient failures.
    """

    def __init__(self,  max_batch_size: int = 50, max_concurrent_batches: int = 5, provider: EmbeddingProvider | None = None):
        self._max_batch_size = max_batch_size
        self._max_concurrent_batches = max_concurrent_batches
        self._provider = provider if provider is not None else OpenAIProvider()

    def batch_chunks(self, chunks: list[str]) -> list[list[str]]:
        """
        Split `chunks` into batches of `_max_batch_size`.
        Step through `chunks` in `_max_batch_size` increments.
        For each increment, slice `chunks` from `i` to `i + _max_batch_size`.
        """
        return [chunks[i:i+self._max_batch_size]
            for i in range(0, len(chunks), self._max_batch_size)]

    @tenacity.retry(
        wait=tenacity.wait_exponential_jitter(initial=1, max=30),
        stop=tenacity.stop_after_attempt(5),
        reraise=True
    )
    async def embed_batch(self, chunk_batch: list[str]) -> list[list[float]]:
        """
        Embed `batch` of `chunks` using `_client`.
        """
        return await self._provider.embed_batch(batch=chunk_batch)

    async def process_chunks(self, chunks: list[str]) -> list[list[float]]:
        """
        Use `_max_batch_size` to generate `batches` of `chunks`.
        Maintain a list of no more than `_max_concurrent_batches` concurrent batches.
            When a batch completes, remove it from the list of concurrent batches,
            add returned embeddings to list of embeddings,
            and add a new `batch` to the list of concurrent batches.
        For each `chunk`, generate an embedding using `_client` where failure is retried with back-off.
        Return list of embeddings.
        """
        if not chunks:
            return []

        # define the semaphore
        semaphore = asyncio.Semaphore(self._max_concurrent_batches)

        # batch chunks
        batches = self.batch_chunks(chunks)

        # for each batch, embed in parallel
        # async with semaphore:
        #   embed_batch()
        async def  embed_batch_with_limited_concurrency(batch):
            async with semaphore:
                return await self.embed_batch(batch)

        batch_results = await asyncio.gather(
            *(embed_batch_with_limited_concurrency(batch) for batch in batches)
        )

        return [embeddings for result in batch_results
                for embeddings in result]


# ----------------------------------

# Example Usage
# =============
# provider = OpenAIProvider()
# embedder = Embedder(provider=provider)
# vectors = await embedder.process_chunks(["abc", "def", ... ])
