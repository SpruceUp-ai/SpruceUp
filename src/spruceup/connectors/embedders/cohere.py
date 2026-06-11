from collections.abc import Callable

import cohere

from ..base import EmbedderConnector, TokenExpiredError


class CohereEmbedder(EmbedderConnector):
    def __init__(
        self,
        api_key: str | Callable[[], str],
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
            self._client = cohere.AsyncClientV2(api_key=self._resolve_api_key())
        return self._client

    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            response = await self._get_client().embed(
                model=self.model,
                texts=batch,
                input_type="search_document",
                embedding_types=["float"],
                output_dimension=self.embedding_dimensions if self._dimensions_overridden else None,
            )
        except cohere.UnauthorizedError as exc:
            raise TokenExpiredError(str(exc)) from exc
        embeddings = response.embeddings.float_
        if embeddings is None:
            raise ValueError("CohereEmbedder: API returned no float embeddings")
        return embeddings
