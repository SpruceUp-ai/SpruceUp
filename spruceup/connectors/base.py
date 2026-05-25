from abc import ABC, abstractmethod

from ..models import ChunkWrapper, SpruceFile, TargetTableConfig


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
    def create_sync_target(self): ...


class EmbedderConfig(ABC):
    @abstractmethod
    def create_provider(self): ...


class SyncTarget(ABC):
    @abstractmethod
    def ensure_table_exists(self, config: TargetTableConfig) -> None: ...

    @abstractmethod
    def sync_batch(
        self,
        upserts: list[ChunkWrapper],
        deletes: list,
        config: TargetTableConfig,
    ) -> None: ...
