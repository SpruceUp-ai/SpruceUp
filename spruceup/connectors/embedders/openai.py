import openai
import tenacity

from ..base import EmbedderConnector


_MODEL_DEFAULT_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


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
        if embedding_dimensions is None:
            if model not in _MODEL_DEFAULT_DIMENSIONS:
                raise ValueError(
                    f"OpenAIEmbedder: unknown model {model!r}; "
                    f"pass embedding_dimensions= explicitly"
                )
            resolved_dimensions = _MODEL_DEFAULT_DIMENSIONS[model]
        else:
            resolved_dimensions = embedding_dimensions
        super().__init__(model=model, api_key=api_key, embedding_dimensions=resolved_dimensions)
        self.max_batch_size = max_batch_size
        self._dimensions_overridden = embedding_dimensions is not None
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            self._client = openai.AsyncOpenAI(api_key=self.api_key)
        return self._client

    @tenacity.retry(
        wait=tenacity.wait_exponential_jitter(initial=1, max=30),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
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
