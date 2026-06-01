from abc import ABC, abstractmethod

from ..models import ChunkWrapper, SpruceFile

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    "txt", "md", "html", "json", "pdf", "doc", "docx",
})


class SourceConnector(ABC):
    @property
    @abstractmethod
    def source_type(self) -> str: ...

    @property
    @abstractmethod
    def source_identifier(self) -> str: ...

    @abstractmethod
    def create_watcher(self, data_source_id: int): ...

    @classmethod
    @abstractmethod
    async def validate(cls, sources: list["SourceConnector"]) -> None: ...

    @abstractmethod
    def is_supported(self, file_identifier: str) -> bool: ...

    @abstractmethod
    async def fetch(self, task) -> "SpruceFile": ...

    @abstractmethod
    def display_name(self, identifier: str) -> str: ...

    @abstractmethod
    def decode_content(self, raw_content: bytes) -> str: ...


class TargetConnector(ABC):
    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @abstractmethod
    def ensure_table_exists(self, embedding_dimensions: int) -> None: ...

    @abstractmethod
    async def sync(self, upserts: list[ChunkWrapper], deletes: list) -> None: ...

    def close(self) -> None: ...


class EmbedderConnector(ABC):
    def __init__(
        self,
        api_key: str | None = None,
        embedding_dimensions: int | None = None,
    ) -> None:
        self.api_key = api_key
        self.embedding_dimensions = embedding_dimensions

    @abstractmethod
    async def embed_batch(self, batch: list[str]) -> list[list[float]]: ...

    async def process_chunks(self, chunks: list[str]) -> list[list[float]]:
        return await self.embed_batch(chunks)

    @property
    def embedding_spec(self) -> str:
        """Canonical "{model}:{dimensions}" string identifying the embedding
        space. Both a model change and a dimensions change yield a new spec, so
        it is the single key the cache and the reindex trigger compare on.

        Default behaviour covers the two shapes in the codebase: a wrapper
        has `_embedder_connector` and delegates to the wrapped lower layer object;
        a concrete API embedder has `_model` + `embedding_dimensions` and builds the spec
        from those fields.
        A subclass with neither must override this."""
        embedder_connector = getattr(self, "_embedder_connector", None)
        if embedder_connector is not None:
            return embedder_connector.embedding_spec
        model = getattr(self, "_model", None)
        if model is not None:
            return f"{model}:{self.embedding_dimensions}"
        raise NotImplementedError(
            f"{type(self).__name__} must define embedding_spec"
        )
