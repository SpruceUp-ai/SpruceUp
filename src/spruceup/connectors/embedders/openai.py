from collections.abc import Callable

import openai

from ..base import EmbedderConnector, TokenExpiredError


class OpenAIEmbedder(EmbedderConnector):
    _client: openai.AsyncOpenAI | None

    def __init__(
        self,
        api_key: str | Callable[[], str],
        model: str = "text-embedding-3-small",
        max_batch_size: int = 150,
        embedding_dimensions: int | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAIEmbedder requires an api_key")
        super().__init__(
            model=model,
            api_key=api_key,
            embedding_dimensions=embedding_dimensions,
            max_batch_size=max_batch_size,
        )
        self._dimensions_overridden = embedding_dimensions is not None

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            self._client = openai.AsyncOpenAI(api_key=self._resolve_api_key())
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        kwargs = {
            "model": self.model,
            "input": batch,
            "encoding_format": "float",
        }
        if self._dimensions_overridden:
            kwargs["dimensions"] = self.embedding_dimensions
        try:
            response = await self._get_client().embeddings.create(**kwargs)
        except openai.AuthenticationError as exc:
            raise TokenExpiredError(str(exc)) from exc
        return [item.embedding for item in response.data]
