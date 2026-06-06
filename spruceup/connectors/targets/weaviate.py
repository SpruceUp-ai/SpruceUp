import asyncio
import typing
import uuid
from typing import Any
from urllib.parse import urlparse

import weaviate
import weaviate.classes as wvc

from ..base import TargetConnector
from ...models import ChunkWrapper
from ...utils.schema import schema_hints, validate_vector_column


_PY_TO_WV: dict[type, Any] = {
    str: wvc.config.DataType.TEXT,
    int: wvc.config.DataType.INT,
    float: wvc.config.DataType.NUMBER,
    bool: wvc.config.DataType.BOOL,
}


def _py_to_wv_type(tp) -> Any | None:
    origin = typing.get_origin(tp)
    if origin is list:
        args = typing.get_args(tp)
        if args == (float,):
            return None
        return wvc.config.DataType.TEXT_ARRAY
    return _PY_TO_WV.get(tp, wvc.config.DataType.TEXT)


class WeaviateTarget(TargetConnector):
    def __init__(
        self,
        collection_name: str,
        schema: type,
        vector_column: str,
        url: str = "http://localhost:8080",
        cluster_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        validate_vector_column(schema, vector_column)
        self.collection_name = collection_name
        self._schema = schema
        self._vector_column = vector_column
        self.url = url
        self.cluster_url = cluster_url
        self.api_key = api_key
        self._client: Any = None
        self._collection: Any = None

    @property
    def display_name(self) -> str:
        return self.collection_name

    @property
    def schema(self) -> type:
        return self._schema

    @property
    def vector_column(self) -> str:
        return self._vector_column

    def identity(self) -> str:
        return f"weaviate:{self.cluster_url or self.url}:{self.collection_name}"

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self.cluster_url:
            auth = wvc.init.Auth.api_key(self.api_key) if self.api_key else None
            self._client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.cluster_url,
                auth_credentials=auth,
            )
        else:
            parsed = urlparse(self.url)
            self._client = weaviate.connect_to_local(
                host=parsed.hostname or "localhost",
                port=parsed.port or 8080,
            )
        return self._client

    def ensure_table_exists(self, embedding_dimensions: int, recreate: bool = False) -> None: # Weaviate doesn't have a dimensions config. It infers it from the first vector inserted.
        client = self._get_client()
        hints = schema_hints(self._schema)

        if recreate and client.collections.exists(self.collection_name):
            client.collections.delete(self.collection_name)

        if not client.collections.exists(self.collection_name):
            properties = [
                wvc.config.Property(name=col, data_type=_py_to_wv_type(tp))
                for col, tp in hints.items()
                if col != self._vector_column and _py_to_wv_type(tp) is not None
            ]
            client.collections.create(
                name=self.collection_name,
                vector_config=wvc.config.Configure.Vectors.self_provided(
                    vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                        distance_metric=wvc.config.VectorDistances.COSINE,
                    )
                ),
                properties=properties,
            )
        self._collection = client.collections.get(self.collection_name)

    @staticmethod
    def _row_uuid(file_id: str, chunk_hash: bytes) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_OID, f"{file_id}:{chunk_hash.hex()}"))

    def _sync_blocking(self, file_id: str, upserts: list[ChunkWrapper], deletes: list[bytes]) -> None:
        collection = self._collection
        hints = schema_hints(self._schema)
        vec_col = self._vector_column

        if deletes:
            collection.data.delete_many(
                where=wvc.query.Filter.by_id().contains_any(
                    [self._row_uuid(file_id, h) for h in deletes]
                )
            )

        if upserts:
            with self._get_client().batch.dynamic() as batch:
                for chunk in upserts:
                    batch.add_object(
                        collection=self.collection_name,
                        uuid=self._row_uuid(file_id, chunk.user_chunk_object_hash),
                        properties={
                            col: getattr(chunk.user_chunk, col)
                            for col in hints
                            if col != vec_col
                        },
                        vector=getattr(chunk.user_chunk, vec_col),
                    )

    async def sync(self, file_id: str, upserts: list[ChunkWrapper], deletes: list[bytes]) -> None:
        await asyncio.to_thread(self._sync_blocking, file_id, upserts, deletes)

    async def aclose(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            self._collection = None
