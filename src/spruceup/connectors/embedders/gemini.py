import math

from google import genai  # pyright: ignore[reportMissingImports]
from google.genai import types  # pyright: ignore[reportMissingImports]

from ..base import EmbedderConnector


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
        # Gemini only returns unit-normalized vectors at the model's native
        # dimension; any reduced output_dimensionality must be normalized here.
        # Normalizing an already-unit native vector is a no-op, so gating on the
        # override alone is safe without knowing the native dimension.
        self._needs_normalization = self._dimensions_overridden
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

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
