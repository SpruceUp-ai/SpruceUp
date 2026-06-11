from collections.abc import Callable
from dataclasses import dataclass

from spruceup.connectors.base import (
    EmbedderConnector,
    SourceConnector,
    TargetConnector,
)
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


class FakeEmbedder(EmbedderConnector):
    def __init__(
        self,
        dimensions: int = 3,
        vector: list[float] | None = None,
        api_key: str | Callable[[], str] | None = None,
    ):
        super().__init__(
            model="fake-model", embedding_dimensions=dimensions, api_key=api_key
        )
        self._vector = vector if vector is not None else [0.0] * dimensions
        self.embedded_batches: list[list[str]] = []

    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        self.embedded_batches.append(list(batch))
        return [list(self._vector) for _ in batch]


class FakeSource(SourceConnector):
    def __init__(self, spruce_file: SpruceFile | None = None, fetch_error: Exception | None = None):
        self._spruce_file = spruce_file
        self._fetch_error = fetch_error
        self.fetched: list[str | None] = []

    @property
    def source_type(self) -> str:
        return "fake"

    @property
    def source_identifier(self) -> str:
        return "fake://source"

    def create_watcher(self, data_source_id: int):
        raise NotImplementedError

    @classmethod
    async def validate(cls, sources) -> None:
        pass

    def is_supported(self, file_identifier: str) -> bool:
        return True

    async def fetch(self, task, manifest) -> SpruceFile:
        self.fetched.append(task.current_file_id)
        if self._fetch_error is not None:
            raise self._fetch_error
        assert self._spruce_file is not None
        return self._spruce_file


def make_transform(n_chunks: int = 2):
    async def transform(*, file_props, embed):
        texts = [f"chunk-{i}" for i in range(n_chunks)]
        vectors = await embed(texts)
        return [DefaultChunk(text=t, embedding=v) for t, v in zip(texts, vectors)]

    return transform


def make_chunk(hash_: bytes, payload=None) -> ChunkWrapper:
    return ChunkWrapper(
        user_chunk=payload if payload is not None else {"hash": hash_.hex()},
        user_chunk_object_hash=hash_,
    )


def make_chunks(n: int) -> list[ChunkWrapper]:
    return [make_chunk(f"hash-{i}".encode()) for i in range(n)]


def make_file(
    file_id: str = "file-1",
    data_source_id: int = 1,
    chunks: list[ChunkWrapper] | None = None,
    modified_at: float = 123.0,
) -> SpruceFile:
    return SpruceFile(
        file_id=file_id,
        display_name="doc.txt",
        file_type="txt",
        data_source_id=data_source_id,
        raw_content=b"hello",
        chunks=chunks or [],
        modified_at=modified_at,
    )
