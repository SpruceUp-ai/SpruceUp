import os

import openai
import tenacity
import dotenv

dotenv.load_dotenv()

from ..base import EmbedderConnector


class OpenAIEmbedder(EmbedderConnector):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        max_batch_size: int = 50,
    ) -> None:
        self._model = model
        self.max_batch_size = max_batch_size
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self._client = openai.AsyncOpenAI(api_key=api_key)
        return self._client

    @tenacity.retry(
        wait=tenacity.wait_exponential_jitter(initial=1, max=30),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        response = await self._get_client().embeddings.create(
            model=self._model,
            input=batch,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]
