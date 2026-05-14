import openai
import tenacity
import asyncio
import os

# Custom embedding provider classes to allow for future extensibility.
class EmbeddingProvider:
    """Abstract base class for embedding providers."""

    def __init__(self, provider: str, model: str, encoding_format: str = "float"):
        self._provider = provider
        self._model = model
        self._encoding_format = encoding_format

    def embed_batch(self, batch: list[str]) -> list[list[float]]:
        pass
        raise NotImplementedError


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider implementation."""

    def __init__(self, model: str = "text-embedding-3-small", encoding_format: str = "float"):
        super().__init__("openai", model, encoding_format)

    def embed_batch(self, batch: list[str]) -> list[list[float]]:
        # should new client be created on each call or reused?
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        pass


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
    model: str
        The model to use for embedding.
    encoding_format: str
        The encoding format to use for the embeddings.
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

    def __init__(self,  max_batch_size: int = 50, max_concurrent_batches: int = 5, provider: EmbeddingProvider = OpenAIProvider):
        self._max_batch_size = max_batch_size
        self._max_concurrent_batches = max_concurrent_batches
        self._provider = provider

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
        stop=tenacity.stop_after_attempt()
    )
    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        """
        Embed `batch` of `chunks` using `_client`.
        """
        pass

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
        # define the semaphore
        semaphore = asyncio.Semaphore(self._max_concurrent_batches)

        # batch chunks
        chunks = self.batch_chunks(chunks)

        # for each batch, embed in parallel
        # async with semaphore:
        #   embed_batch()
        pass
