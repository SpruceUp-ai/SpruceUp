import voyageai
import tenacity

from ..base import EmbedderConnector


_MODEL_DEFAULT_DIMENSIONS: dict[str, int] = {
    "voyage-4-large": 1024,
    "voyage-4-lite": 1024,
    "voyage-3-large": 1024,
    "voyage-3": 1024,
    "voyage-3-lite": 512,
}

_VOYAGE_4_ALLOWED_DIMENSIONS = {256, 512, 1024, 2048}


class VoyageAIEmbedder(EmbedderConnector):
    def __init__(
        self,
        api_key: str,
        model: str = "voyage-4-large",
        max_batch_size: int = 150,
        embedding_dimensions: int | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("VoyageAIEmbedder requires an api_key")
        if embedding_dimensions is None:
            if model not in _MODEL_DEFAULT_DIMENSIONS:
                raise ValueError(
                    f"VoyageAIEmbedder: unknown model {model!r}; "
                    f"pass embedding_dimensions= explicitly"
                )
            resolved_dimensions = _MODEL_DEFAULT_DIMENSIONS[model]
        else:
            if (
                model.startswith("voyage-4")
                and embedding_dimensions not in _VOYAGE_4_ALLOWED_DIMENSIONS
            ):
                raise ValueError(
                    f"VoyageAIEmbedder: model {model!r} only supports "
                    f"{sorted(_VOYAGE_4_ALLOWED_DIMENSIONS)}-dim outputs; "
                    f"got {embedding_dimensions}"
                )
            resolved_dimensions = embedding_dimensions
        super().__init__(
            model=model,
            api_key=api_key,
            embedding_dimensions=resolved_dimensions,
            max_batch_size=max_batch_size,
        )
        self._client: voyageai.AsyncClient | None = None

    def _get_client(self) -> voyageai.AsyncClient:
        if self._client is None:
            self._client = voyageai.AsyncClient(api_key=self.api_key)
        return self._client

    @tenacity.retry(
        wait=tenacity.wait_exponential_jitter(initial=1, max=30),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        response = await self._get_client().embed(
            texts=batch,
            model=self.model,
            output_dimension=self.embedding_dimensions,
        )
        return response.embeddings
