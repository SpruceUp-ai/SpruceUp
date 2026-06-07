import math

import tenacity
from google import genai  # pyright: ignore[reportMissingImports]
from google.genai import types  # pyright: ignore[reportMissingImports]

from ..base import EmbedderConnector


_MODEL_DEFAULT_DIMENSIONS: dict[str, int] = {
    "gemini-embedding-001": 3072,
    "text-embedding-004": 768,
    "embedding-001": 768,
}


_NATIVE_DIMENSIONS: dict[str, int] = {
    "gemini-embedding-001": 3072,
    "text-embedding-004": 768,
    "embedding-001": 768,
}


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class GeminiEmbedder(EmbedderConnector):
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        max_batch_size: int = 100,
        embedding_dimensions: int | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GeminiEmbedder requires an api_key")
        if model not in _MODEL_DEFAULT_DIMENSIONS:
            raise ValueError(
                f"GeminiEmbedder: unknown model {model!r}; "
                f"supported: {sorted(_MODEL_DEFAULT_DIMENSIONS)}"
            )
        if max_batch_size > 100:
            raise ValueError(
                f"GeminiEmbedder: max_batch_size cannot exceed 100 (API limit); "
                f"got {max_batch_size}"
            )
        resolved_dimensions = embedding_dimensions or _MODEL_DEFAULT_DIMENSIONS[model]
        super().__init__(
            model=model,
            api_key=api_key,
            embedding_dimensions=resolved_dimensions,
            max_batch_size=max_batch_size,
        )
        self._dimensions_overridden = embedding_dimensions is not None
        self._needs_normalization = (
            resolved_dimensions != _NATIVE_DIMENSIONS[model]
        )
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @tenacity.retry(
        wait=tenacity.wait_exponential_jitter(initial=1, max=30),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        config_kwargs = {"task_type": "RETRIEVAL_DOCUMENT"}
        if self._dimensions_overridden:
            config_kwargs["output_dimensionality"] = self.embedding_dimensions
        response = await self._get_client().aio.models.embed_content(
            model=self.model,
            contents=batch,
            config=types.EmbedContentConfig(**config_kwargs),
        )
        vectors = [item.values for item in response.embeddings]
        if self._needs_normalization:
            vectors = [_l2_normalize(v) for v in vectors]
        return vectors
