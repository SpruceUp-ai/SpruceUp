from abc import ABC, abstractmethod


class SourceConnector(ABC):
    @abstractmethod
    def create_watcher(self): ...


class TargetConnector(ABC):
    @abstractmethod
    def create_sync_target(self): ...


class EmbedderConfig(ABC):
    @abstractmethod
    def create_provider(self): ...
