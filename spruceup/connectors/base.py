from abc import ABC, abstractmethod

from ..models import ChunkWrapper, SpruceFile


class EmbeddingError(Exception):
    """Raised when the embedding API fails after all retries are exhausted."""

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    "txt", "md", "html", "json", "pdf", "doc", "docx",
})


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

    @abstractmethod
    def identifier_from_file_id(self, file_id: str) -> str: ...

    @abstractmethod
    def display_name(self, identifier: str) -> str: ...

    @abstractmethod
    def decode_content(self, raw_content: bytes) -> str: ...


class TargetConnector(ABC):
    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @property
    @abstractmethod
    def schema(self) -> type: ...

    @abstractmethod
    def ensure_table_exists(self, embedding_dimensions: int) -> None: ...

    @abstractmethod
    async def sync(self, upserts: list[ChunkWrapper], deletes: list[bytes]) -> None: ...

    async def aclose(self) -> None: ...


class EmbedderConnector(ABC):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        embedding_dimensions: int | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.embedding_dimensions = embedding_dimensions

    @abstractmethod
    async def embed_batch(self, batch: list[str]) -> list[list[float]]: ...

    async def process_chunks(self, chunks: list[str]) -> list[list[float]]:
        return await self.embed_batch(chunks)

    async def aclose(self) -> None: ...
