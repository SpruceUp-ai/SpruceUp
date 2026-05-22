from abc import ABC, abstractmethod


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
