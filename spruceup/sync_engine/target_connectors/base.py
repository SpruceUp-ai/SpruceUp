from abc import ABC, abstractmethod

from ...models import ChunkWrapper, TargetTableConfig


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
