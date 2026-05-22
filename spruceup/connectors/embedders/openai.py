from dataclasses import dataclass

from ..base import EmbedderConfig


@dataclass
class OpenAIEmbedder(EmbedderConfig):
    model: str = "text-embedding-3-small"

    def create_provider(self):
        from spruceup.embedding import OpenAIProvider
        return OpenAIProvider(model=self.model)
