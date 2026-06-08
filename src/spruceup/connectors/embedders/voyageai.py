import voyageai

from ..base import EmbedderConnector


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
        super().__init__(
            model=model,
            api_key=api_key,
            embedding_dimensions=embedding_dimensions,
            max_batch_size=max_batch_size,
        )
        self._dimensions_overridden = embedding_dimensions is not None
        self._client: voyageai.AsyncClient | None = None

    def _get_client(self) -> voyageai.AsyncClient:
        if self._client is None:
            self._client = voyageai.AsyncClient(api_key=self.api_key)
        return self._client

    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        kwargs = {"texts": batch, "model": self.model}
        if self._dimensions_overridden:
            kwargs["output_dimension"] = self.embedding_dimensions
        response = await self._get_client().embed(**kwargs)
        return response.embeddings
