from dataclasses import dataclass

from ..base import TargetConnector


@dataclass
class PgVectorTarget(TargetConnector):
    connstr: str
    table: str
    schema: type
    primary_key: str

    def create_sync_target(self):
        from spruceup.sync_engine.target_connectors.pgvector import PgVectorSyncTarget
        return PgVectorSyncTarget(self.connstr)
