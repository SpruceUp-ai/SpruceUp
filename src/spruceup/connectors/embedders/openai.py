import openai

from ..base import EmbedderConnector


class OpenAIEmbedder(EmbedderConnector):
    def __init__(
        self,
        api_key: str,
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
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            self._client = openai.AsyncOpenAI(api_key=self.api_key)
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
        response = await self._get_client().embeddings.create(**kwargs)
        return [item.embedding for item in response.data]
