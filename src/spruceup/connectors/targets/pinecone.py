import asyncio
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from ..base import TargetConnector
from ...models import ChunkWrapper
from ...utils.schema import schema_hints


class PineconeTarget(TargetConnector):
    def __init__(
        self,
        api_key: str | None,
        index_name: str,
        schema: type,
        vector_column: str,
        namespace: str = "",
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        super().__init__(schema, vector_column)
        self.api_key = api_key
        self.index_name = index_name
        self.namespace = namespace
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self._pc: Any = None
        self._index: Any = None

    @property
    def display_name(self) -> str:
        return self.index_name

    def identity(self) -> str:
        return f"pinecone:{self.index_name}:{self.namespace}"

    def _client(self) -> Any:
        if self._pc is None:
            self._pc = Pinecone(api_key=self.api_key)
        return self._pc

    def ensure_table_exists(self, embedding_dimensions: int, recreate: bool = False) -> None:
        pc = self._client()
        exists = self.index_name in pc.list_indexes().names()
        if recreate and exists:
            pc.delete_index(self.index_name)
            exists = False
        if not exists:
            pc.create_index(
                name=self.index_name,
                dimension=embedding_dimensions,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
        self._index = pc.Index(self.index_name)

    async def sync(self, file_id: str, upserts: list[ChunkWrapper], deletes: list[bytes]) -> None:
        index = self._index
        hints = schema_hints(self._schema)
        vector_col = self._vector_column

        if upserts:
            vectors = [
                {
                    "id": f"{file_id}:{chunk.user_chunk_object_hash.hex()}",
                    "values": getattr(chunk.user_chunk, vector_col),
                    "metadata": {
                        col: getattr(chunk.user_chunk, col)
                        for col in hints
                        if col != vector_col
                    },
                }
                for chunk in upserts
            ]
            await asyncio.to_thread(index.upsert, vectors=vectors, namespace=self.namespace)

        if deletes:
            await asyncio.to_thread(index.delete, ids=[f"{file_id}:{h.hex()}" for h in deletes], namespace=self.namespace)
