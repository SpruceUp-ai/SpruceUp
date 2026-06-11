import math
from collections.abc import Callable
from typing import Any, cast

from google import genai  # pyright: ignore[reportMissingImports]
from google.genai import errors, types  # pyright: ignore[reportMissingImports]

from ..base import EmbedderConnector, TokenExpiredError


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class GeminiEmbedder(EmbedderConnector):
    def __init__(
        self,
        api_key: str | Callable[[], str],
        model: str = "gemini-embedding-001",
        max_batch_size: int = 100,
        embedding_dimensions: int | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GeminiEmbedder requires an api_key")
        if max_batch_size > 100:
            raise ValueError(
                f"GeminiEmbedder: max_batch_size cannot exceed 100 (API limit); "
                f"got {max_batch_size}"
            )
        super().__init__(
            model=model,
            api_key=api_key,
            embedding_dimensions=embedding_dimensions,
            max_batch_size=max_batch_size,
        )
        self._dimensions_overridden = embedding_dimensions is not None
        # Reduced-dimension outputs aren't unit-normalized by the API; fix here.
        self._needs_normalization = self._dimensions_overridden
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self._resolve_api_key())
        return self._client

    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            response = await self._get_client().aio.models.embed_content(
                model=self.model,
                contents=cast(Any, batch),
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=(
                        self.embedding_dimensions if self._dimensions_overridden else None
                    ),
                ),
            )
        except errors.ClientError as exc:
            if exc.code in (401, 403):
                raise TokenExpiredError(str(exc)) from exc
            raise
        vectors: list[list[float]] = []
        for item in response.embeddings or []:
            if item.values is None:
                raise ValueError("GeminiEmbedder: API returned an empty embedding")
            vectors.append(item.values)
        if self._needs_normalization:
            vectors = [_l2_normalize(v) for v in vectors]
        return vectors
