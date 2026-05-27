import typing
from dataclasses import dataclass, field
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from ..base import TargetConnector
from ...models import ChunkWrapper


def _vector_field(hints: dict) -> str:
    for col, tp in hints.items():
        origin = typing.get_origin(tp)
        if origin is list:
            args = typing.get_args(tp)
            if args == (float,):
                return col
    raise ValueError("Schema has no list[float] field for vector values")


@dataclass
class PineconeTarget(TargetConnector):
    api_key: str | None
    index_name: str
    schema: type
    primary_key: str
    namespace: str = ""
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"
    _pc: Any = field(default=None, init=False, repr=False)
    _index: Any = field(default=None, init=False, repr=False)

    @property
    def display_name(self) -> str:
        return self.index_name

    def _client(self) -> Any:
        if self._pc is None:
            self._pc = Pinecone(api_key=self.api_key)
        return self._pc

    def ensure_table_exists(self, embedding_dimensions: int) -> None:
        pc = self._client()
        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.index_name,
                dimension=embedding_dimensions,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
        self._index = pc.Index(self.index_name)

    def sync(self, upserts: list[ChunkWrapper], deletes: list) -> None:
        index = self._index
        hints = typing.get_type_hints(self.schema)
        vector_col = _vector_field(hints)

        if upserts:
            vectors = [
                {
                    "id": str(getattr(chunk.user_chunk, self.primary_key)),
                    "values": getattr(chunk.user_chunk, vector_col),
                    "metadata": {
                        col: getattr(chunk.user_chunk, col)
                        for col in hints
                        if col != self.primary_key and col != vector_col
                    },
                }
                for chunk in upserts
            ]
            index.upsert(vectors=vectors, namespace=self.namespace)

        if deletes:
            index.delete(ids=[str(d) for d in deletes], namespace=self.namespace)
