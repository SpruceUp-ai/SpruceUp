import asyncio
import typing
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import weaviate
import weaviate.classes as wvc

from ..base import TargetConnector
from ...models import ChunkWrapper


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


def _vector_field(hints: dict) -> str:
    for col, tp in hints.items():
        if typing.get_origin(tp) is list and typing.get_args(tp) == (float,):
            return col
    raise ValueError("Schema has no list[float] field for the embedding vector")


@dataclass
class WeaviateTarget(TargetConnector):
    collection_name: str
    schema: type
    primary_key: str
    url: str = "http://localhost:8080"
    cluster_url: str | None = None 
    api_key: str | None = None
    _client: Any = field(default=None, init=False, repr=False)
    _collection: Any = field(default=None, init=False, repr=False)

    @property
    def display_name(self) -> str:
        return self.collection_name

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

    def ensure_table_exists(self, embedding_dimensions: int) -> None: # Weaviate doesn't have a dimensions config. It infers it from the first vector inserted.
        client = self._get_client()
        hints = typing.get_type_hints(self.schema)
        _vector_field(hints)  # validate a vector field exists before creating the collection

        if not client.collections.exists(self.collection_name):
            properties = [
                wvc.config.Property(name=col, data_type=_py_to_wv_type(tp))
                for col, tp in hints.items()
                if _py_to_wv_type(tp) is not None
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

    def _sync_blocking(self, upserts: list[ChunkWrapper], deletes: list) -> None:
        collection = self._collection
        hints = typing.get_type_hints(self.schema)
        vec_col = _vector_field(hints)

        if deletes:
            collection.data.delete_many(
                where=wvc.query.Filter.by_id().contains_any(deletes)
            )

        if upserts:
            with self._get_client().batch.dynamic() as batch:
                for chunk in upserts:
                    pk_val = getattr(chunk.user_chunk, self.primary_key)
                    batch.add_object(
                        collection=self.collection_name,
                        uuid=pk_val,
                        properties={
                            col: getattr(chunk.user_chunk, col)
                            for col in hints
                            if col != vec_col
                        },
                        vector=getattr(chunk.user_chunk, vec_col),
                    )

    async def sync(self, upserts: list[ChunkWrapper], deletes: list) -> None:
        await asyncio.to_thread(self._sync_blocking, upserts, deletes)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            self._collection = None
