import cohere

from ..base import EmbedderConnector


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
        super().__init__(
            model=model,
            api_key=api_key,
            embedding_dimensions=embedding_dimensions,
            max_batch_size=max_batch_size,
        )
        self._dimensions_overridden = embedding_dimensions is not None
        self._client: cohere.AsyncClientV2 | None = None

    def _get_client(self) -> cohere.AsyncClientV2:
        if self._client is None:
            self._client = cohere.AsyncClientV2(api_key=self.api_key)
        return self._client

    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        kwargs = {
            "model": self.model,
            "texts": batch,
            "input_type": "search_document",
            "embedding_types": ["float"],
        }
        if self._dimensions_overridden:
            kwargs["output_dimension"] = self.embedding_dimensions
        response = await self._get_client().embed(**kwargs)
        return response.embeddings.float_
