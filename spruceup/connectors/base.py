from abc import ABC, abstractmethod

from ..models import ChunkWrapper, SpruceFile


class SourceConnector(ABC):
    @property
    @abstractmethod
    def source_type(self) -> str: ...

    @property
    @abstractmethod
    def source_identifier(self) -> str: ...

    @abstractmethod
    def create_watcher(self, data_source_id: int): ...

    @abstractmethod
    async def fetch(self, task) -> "SpruceFile": ...

    @abstractmethod
    def display_name(self, identifier: str) -> str: ...

    @abstractmethod
    def decode_content(self, raw_content: bytes) -> str: ...


class TargetConnector(ABC):
    @abstractmethod
    def ensure_table_exists(self) -> None: ...

    @abstractmethod
    def sync(self, upserts: list[ChunkWrapper], deletes: list) -> None: ...


class EmbedderConnector(ABC):
    @abstractmethod
    async def embed_batch(self, batch: list[str]) -> list[list[float]]: ...

    def expect(self, n: int = 1) -> None:
        """Hint that n more process_chunks calls are coming. Default no-op."""
        pass
