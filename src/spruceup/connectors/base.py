from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import tenacity

from ..models import ChunkWrapper, SpruceFile
from ..utils.schema import validate_vector_column

if TYPE_CHECKING:
    from ..manifest import Manifest
    from ..models import SyncTask
    from ..monitoring.monitor import BaseWatcher


class EmbeddingError(Exception):
    """Raised when the embedding API fails after all retries are exhausted."""


class EmbeddingConfigError(Exception):
    """Raised when the embedder's model, credentials, or dimensions don't match
    what the provider's API actually accepts/returns (detected by health_check)."""


class TokenExpiredError(Exception):
    """Raised by an embedder when the API rejects its credentials, signalling that
    a callable api_key should be re-resolved and the client rebuilt before retry."""

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    "txt", "md", "html", "json", "pdf", "doc", "docx",
})

BINARY_EXTENSIONS: frozenset[str] = frozenset({"pdf", "doc", "docx"})


class SourceConnector(ABC):
    @property
    @abstractmethod
    def source_type(self) -> str: ...

    @property
    @abstractmethod
    def source_identifier(self) -> str: ...

    @abstractmethod
    def create_watcher(self, data_source_id: int) -> "BaseWatcher": ...

    @classmethod
    @abstractmethod
    async def validate(cls, sources: list["SourceConnector"]) -> None: ...

    @abstractmethod
    def is_supported(self, file_identifier: str) -> bool: ...

    @abstractmethod
    async def fetch(self, task: "SyncTask", manifest: "Manifest") -> "SpruceFile":
        # modified_at must be set to a Unix timestamp (float seconds).
        ...

    def decode_content(self, raw_content: bytes, file_type: str) -> str | bytes:
        if file_type.lower() in BINARY_EXTENSIONS:
            return raw_content
        return raw_content.decode("utf-8", errors="replace")


class TargetConnector(ABC):
    def __init__(self, schema: type, vector_column: str) -> None:
        validate_vector_column(schema, vector_column)
        self._schema = schema
        self._vector_column = vector_column

    @property
    def schema(self) -> type:
        return self._schema

    @property
    def vector_column(self) -> str:
        return self._vector_column

    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @abstractmethod
    def identity(self) -> str:
        # Must exclude credentials so rotating a password doesn't trigger a reindex.
        ...

    @abstractmethod
    def ensure_table_exists(self, embedding_dimensions: int, recreate: bool = False) -> None: ...

    @abstractmethod
    async def sync(self, file_id: str, upserts: list[ChunkWrapper], deletes: list[bytes]) -> None: ...

    async def aclose(self) -> None: ...


class EmbedderConnector(ABC):
    def __init__(
        self,
        model: str,
        api_key: str | Callable[[], str] | None = None,
        embedding_dimensions: int | None = None,
        max_batch_size: int = 100,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.embedding_dimensions = embedding_dimensions
        self.max_batch_size = max_batch_size
        self._client: Any = None

    def _resolve_api_key(self) -> str | None:
        return self.api_key() if callable(self.api_key) else self.api_key

    def _invalidate_client(self) -> None:
        self._client = None

    @abstractmethod
    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        # Raw API call, no retry (retry lives in embed_batch_retrying).
        ...

    async def embed_batch_retrying(self, batch: list[str]) -> list[list[float]]:
        async for attempt in tenacity.AsyncRetrying(
            wait=tenacity.wait_exponential_jitter(initial=1, max=30),
            stop=tenacity.stop_after_attempt(5),
            before_sleep=self._refresh_token_before_retry,
            reraise=True,
        ):
            with attempt:
                return await self.embed_batch(batch)
        raise AssertionError("unreachable: reraise=True returns or raises")

    def _refresh_token_before_retry(self, retry_state: tenacity.RetryCallState) -> None:
        outcome = retry_state.outcome
        if outcome is None:
            return
        if isinstance(outcome.exception(), TokenExpiredError) and callable(self.api_key):
            self._invalidate_client()

    async def process_chunks(self, chunks: list[str]) -> list[list[float]]:
        return await self.embed_batch_retrying(chunks)

    async def health_check(self) -> None:
        try:
            vectors = await self.embed_batch(["spruceup embedder health check"])
        except Exception as exc:
            raise EmbeddingConfigError(
                f"{type(self).__name__}: health check failed for model "
                f"{self.model!r} — {exc}"
            ) from exc
        if not vectors or not vectors[0]:
            raise EmbeddingConfigError(
                f"{type(self).__name__}: model {self.model!r} returned an empty "
                f"embedding during health check"
            )
        actual = len(vectors[0])
        if self.embedding_dimensions is None:
            self.embedding_dimensions = actual
        elif self.embedding_dimensions != actual:
            raise EmbeddingConfigError(
                f"{type(self).__name__}: configured embedding_dimensions="
                f"{self.embedding_dimensions} but model {self.model!r} returned "
                f"{actual}-dim vectors"
            )

    async def aclose(self) -> None: ...
