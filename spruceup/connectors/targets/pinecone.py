import typing
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


class PineconeTarget(TargetConnector):
    def __init__(
        self,
        api_key: str | None,
        index_name: str,
        schema: type,
        namespace: str = "",
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        self.api_key = api_key
        self.index_name = index_name
        self._schema = schema
        self.namespace = namespace
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self._pc: Any = None
        self._index: Any = None

    @property
    def display_name(self) -> str:
        return self.index_name

    @property
    def schema(self) -> type:
        return self._schema

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

    async def sync(self, upserts: list[ChunkWrapper], deletes: list[bytes]) -> None:
        index = self._index
        hints = typing.get_type_hints(self._schema)
        vector_col = _vector_field(hints)

        if upserts:
            vectors = [
                {
                    "id": chunk.user_chunk_object_hash.hex(),
                    "values": getattr(chunk.user_chunk, vector_col),
                    "metadata": {
                        col: getattr(chunk.user_chunk, col)
                        for col in hints
                        if col != vector_col
                    },
                }
                for chunk in upserts
            ]
            index.upsert(vectors=vectors, namespace=self.namespace)

        if deletes:
            index.delete(ids=[h.hex() for h in deletes], namespace=self.namespace)
