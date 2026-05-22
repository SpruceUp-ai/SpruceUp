from dataclasses import dataclass

from .connectors.base import EmbedderConfig, SourceConnector, TargetConnector


@dataclass
class SpruceUpConfig:
    sources: list[SourceConnector]
    target: TargetConnector
    embeddings: EmbedderConfig


def defineConfig(*, sources, target, embeddings) -> SpruceUpConfig:
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list of source connectors")
    for i, source in enumerate(sources):
        if not isinstance(source, SourceConnector):
            raise TypeError(
                f"sources[{i}] must be a SourceConnector, got {type(source).__name__!r}"
            )

    if not isinstance(target, TargetConnector):
        raise TypeError(f"target must be a TargetConnector, got {type(target).__name__!r}")

    if not isinstance(embeddings, EmbedderConfig):
        raise TypeError(
            f"embeddings must be an EmbedderConfig, got {type(embeddings).__name__!r}"
        )

    return SpruceUpConfig(sources=sources, target=target, embeddings=embeddings)
