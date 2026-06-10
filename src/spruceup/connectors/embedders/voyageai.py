from collections.abc import Callable
from typing import cast

from voyageai import AsyncClient  # pyright: ignore[reportPrivateImportUsage]
from voyageai.error import AuthenticationError

from ..base import EmbedderConnector, TokenExpiredError


class VoyageAIEmbedder(EmbedderConnector):
    def __init__(
        self,
        api_key: str | Callable[[], str],
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
        self._client: AsyncClient | None = None

    def _get_client(self) -> AsyncClient:
        if self._client is None:
            self._client = AsyncClient(api_key=self._resolve_api_key())
        return self._client

    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            response = await self._get_client().embed(
                texts=batch,
                model=self.model,
                output_dimension=self.embedding_dimensions if self._dimensions_overridden else None,
            )
        except AuthenticationError as exc:
            raise TokenExpiredError(str(exc)) from exc
        return cast(list[list[float]], response.embeddings)
