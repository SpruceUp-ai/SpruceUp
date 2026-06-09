from dataclasses import dataclass

from spruceup.connectors.base import TargetConnector
from spruceup.models import ChunkWrapper, SpruceFile


@dataclass
class DefaultChunk:
    text: str
    embedding: list[float]


class FakeTarget(TargetConnector):
    def __init__(self, schema: type = DefaultChunk, vector_column: str = "embedding"):
        super().__init__(schema, vector_column)
        self.calls: list[tuple[str, list[ChunkWrapper], list[bytes]]] = []

    @property
    def display_name(self) -> str:
        return "fake-target"

    def identity(self) -> str:
        return "fake://target"

    def ensure_table_exists(self, embedding_dimensions: int, recreate: bool = False) -> None:
        pass

    async def sync(
        self, file_id: str, upserts: list[ChunkWrapper], deletes: list[bytes]
    ) -> None:
        self.calls.append((file_id, list(upserts), list(deletes)))


def make_chunk(hash_: bytes, payload=None) -> ChunkWrapper:
    return ChunkWrapper(
        user_chunk=payload if payload is not None else {"hash": hash_.hex()},
        user_chunk_object_hash=hash_,
    )


def make_file(
    file_id: str = "file-1",
    data_source_id: int = 1,
    chunks: list[ChunkWrapper] | None = None,
) -> SpruceFile:
    return SpruceFile(
        file_id=file_id,
        display_name="doc.txt",
        file_type="txt",
        data_source_id=data_source_id,
        raw_content=b"hello",
        chunks=chunks or [],
        modified_at=123.0,
    )
