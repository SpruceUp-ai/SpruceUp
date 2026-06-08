from abc import ABC, abstractmethod

import tenacity

from ..models import ChunkWrapper, SpruceFile


class EmbeddingError(Exception):
    """Raised when the embedding API fails after all retries are exhausted."""


class EmbeddingConfigError(Exception):
    """Raised when the embedder's model, credentials, or dimensions don't match
    what the provider's API actually accepts/returns (detected by health_check)."""

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    "txt", "md", "html", "json", "pdf", "doc", "docx",
})

# Formats whose parsers need the original bytes; a utf-8 decode would corrupt
# them, so decode_content passes these through unchanged. doc/docx here are
# Microsoft Word files — Google Docs are exported to text upstream and arrive
# as "txt", so they are never treated as binary.
BINARY_EXTENSIONS: frozenset[str] = frozenset({"pdf", "doc", "docx"})


class SourceConnector(ABC):
    @property
    @abstractmethod
    def source_type(self) -> str: ...

    @property
    @abstractmethod
    def source_identifier(self) -> str: ...

    @abstractmethod
    def create_watcher(self, data_source_id: int): ...

    @classmethod
    @abstractmethod
    async def validate(cls, sources: list["SourceConnector"]) -> None: ...

    @abstractmethod
    def is_supported(self, file_identifier: str) -> bool: ...

    @abstractmethod
    async def fetch(self, task, manifest) -> "SpruceFile":
        """Fetch a file and return a SpruceFile.

        Implementations must populate SpruceFile.modified_at with a Unix
        timestamp (seconds since epoch, float). Convert from whatever format
        the source provides natively (ISO 8601, datetime, etc.) at this boundary.
        """
        ...

    def decode_content(self, raw_content: bytes, file_type: str) -> str | bytes:
        """Prepare raw bytes for the transform function.

        Binary formats (see BINARY_EXTENSIONS) are returned unchanged so the
        user's parser receives the original bytes it expects. Text formats are
        decoded to a utf-8 str.
        """
        if file_type.lower() in BINARY_EXTENSIONS:
            return raw_content
        return raw_content.decode("utf-8", errors="replace")


class TargetConnector(ABC):
    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @property
    @abstractmethod
    def schema(self) -> type: ...

    @property
    @abstractmethod
    def vector_column(self) -> str:
        """Name of the schema field holding the embedding vector."""
        ...

    @abstractmethod
    def identity(self) -> str:
        """Stable identity of this target for change detection.

        Must exclude credentials (so rotating a password does not trigger a
        reindex) but capture anything whose change means a different physical
        destination (host, database, table/index/collection).
        """
        ...

    @abstractmethod
    def ensure_table_exists(self, embedding_dimensions: int, recreate: bool = False) -> None:
        """Create the target table/index. If recreate, drop it first."""
        ...

    @abstractmethod
    async def sync(self, file_id: str, upserts: list[ChunkWrapper], deletes: list[bytes]) -> None: ...

    async def aclose(self) -> None: ...


class EmbedderConnector(ABC):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        embedding_dimensions: int | None = None,
        max_batch_size: int = 100,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.embedding_dimensions = embedding_dimensions
        self.max_batch_size = max_batch_size

    @abstractmethod
    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Raw embedding API call for one batch — no retry.

        The shared transient-failure retry policy lives in embed_batch_retrying,
        which the production path uses. Keeping embed_batch raw lets health_check
        fail fast on a bad model/credential/dimension instead of retrying it.
        """
        ...

    async def embed_batch_retrying(self, batch: list[str]) -> list[list[float]]:
        async for attempt in tenacity.AsyncRetrying(
            wait=tenacity.wait_exponential_jitter(initial=1, max=30),
            stop=tenacity.stop_after_attempt(5),
            reraise=True,
        ):
            with attempt:
                return await self.embed_batch(batch)

    async def process_chunks(self, chunks: list[str]) -> list[list[float]]:
        return await self.embed_batch_retrying(chunks)

    async def health_check(self) -> None:
        """Validate model name, credentials, and dimensions against the live API.

        Embeds a short probe string. A wrong model name, bad credentials, or an
        unsupported embedding_dimensions surfaces as the provider's own API error,
        re-raised as EmbeddingConfigError. The probe's vector length resolves
        embedding_dimensions when the user left it unset, or is checked against the
        user's value when set. Runs once at startup, before the target table is
        created (which needs the dimension) and before reindex fingerprinting.
        """
        try:
            vectors = await self.embed_batch(["spruceup embedder health check"])
        except Exception as exc:
            raise EmbeddingConfigError(
                f"{type(self).__name__}: health check failed for model "
                f"{self.model!r} — {exc}"
            ) from exc
        if not vectors or not vectors[0]:
            raise EmbeddingConfigError(
                f"{type(self).__name__}: model {self.model!r} returned an empty "
                f"embedding during health check"
            )
        actual = len(vectors[0])
        if self.embedding_dimensions is None:
            self.embedding_dimensions = actual
        elif self.embedding_dimensions != actual:
            raise EmbeddingConfigError(
                f"{type(self).__name__}: configured embedding_dimensions="
                f"{self.embedding_dimensions} but model {self.model!r} returned "
                f"{actual}-dim vectors"
            )

    async def aclose(self) -> None: ...
