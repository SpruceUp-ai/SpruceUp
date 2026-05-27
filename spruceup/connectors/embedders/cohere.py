import cohere
import tenacity

from ..base import EmbedderConnector


_MODEL_DEFAULT_DIMENSIONS: dict[str, int] = {
    "embed-v4.0": 1536,
    "embed-english-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-v3.0": 1024,
    "embed-multilingual-light-v3.0": 384,
}


_EMBED_V4_ALLOWED_DIMENSIONS = {256, 512, 1024, 1536}


class CohereEmbedder(EmbedderConnector):
    def __init__(
        self,
        api_key: str,
        model: str = "embed-v4.0",
        max_batch_size: int = 96,
        embedding_dimensions: int | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("CohereEmbedder requires an api_key")
        if embedding_dimensions is None:
            if model not in _MODEL_DEFAULT_DIMENSIONS:
                raise ValueError(
                    f"CohereEmbedder: unknown model {model!r}; "
                    f"pass embedding_dimensions= explicitly"
                )
            resolved_dimensions = _MODEL_DEFAULT_DIMENSIONS[model]
        else:
            if (
                model.startswith("embed-v4")
                and embedding_dimensions not in _EMBED_V4_ALLOWED_DIMENSIONS
            ):
                raise ValueError(
                    f"CohereEmbedder: model {model!r} only supports "
                    f"{sorted(_EMBED_V4_ALLOWED_DIMENSIONS)}-dim outputs; "
                    f"got {embedding_dimensions}"
                )
            resolved_dimensions = embedding_dimensions
        super().__init__(api_key=api_key, embedding_dimensions=resolved_dimensions)
        self._model = model
        self.max_batch_size = max_batch_size
        self._dimensions_overridden = embedding_dimensions is not None
        self._client: cohere.AsyncClientV2 | None = None

    def _get_client(self) -> cohere.AsyncClientV2:
        if self._client is None:
            self._client = cohere.AsyncClientV2(api_key=self.api_key)
        return self._client

    @tenacity.retry(
        wait=tenacity.wait_exponential_jitter(initial=1, max=30),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        kwargs = {
            "model": self._model,
            "texts": batch,
            "input_type": "search_document",
            "embedding_types": ["float"],
        }
        if self._dimensions_overridden:
            kwargs["output_dimension"] = self.embedding_dimensions
        response = await self._get_client().embed(**kwargs)
        return response.embeddings.float_
