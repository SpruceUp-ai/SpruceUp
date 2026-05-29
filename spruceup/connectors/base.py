from abc import ABC, abstractmethod

from ..models import ChunkWrapper, SpruceFile

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
    async def fetch(self, task) -> "SpruceFile": ...

    @abstractmethod
    def display_name(self, identifier: str) -> str: ...

    @abstractmethod
    def decode_content(self, raw_content: bytes) -> str: ...


class TargetConnector(ABC):
    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @abstractmethod
    def ensure_table_exists(self, embedding_dimensions: int) -> None: ...

    @abstractmethod
    async def sync(self, upserts: list[ChunkWrapper], deletes: list) -> None: ...


class EmbedderConnector(ABC):
    def __init__(
        self,
        api_key: str | None = None,
        embedding_dimensions: int | None = None,
    ) -> None:
        self.api_key = api_key
        self.embedding_dimensions = embedding_dimensions

    @abstractmethod
    async def embed_batch(self, batch: list[str]) -> list[list[float]]: ...

    async def process_chunks(self, chunks: list[str]) -> list[list[float]]:
        return await self.embed_batch(chunks)
