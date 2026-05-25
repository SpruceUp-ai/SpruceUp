from dataclasses import dataclass
from typing import Callable

from .connectors.base import EmbedderConnector, SourceConnector, TargetConnector


@dataclass
class SpruceUpConfig:
    sources: list[SourceConnector]
    target: TargetConnector
    embeddings: EmbedderConnector
    transform: Callable


def defineConfig(*, sources, target, embeddings, transform) -> SpruceUpConfig:
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list of source connectors")
    for i, source in enumerate(sources):
        if not isinstance(source, SourceConnector):
            raise TypeError(
                f"sources[{i}] must be a SourceConnector, got {type(source).__name__!r}"
            )

    if not isinstance(target, TargetConnector):
        raise TypeError(f"target must be a TargetConnector, got {type(target).__name__!r}")

    if not isinstance(embeddings, EmbedderConnector):
        raise TypeError(
            f"embeddings must be an EmbedderConnector, got {type(embeddings).__name__!r}"
        )

    if not callable(transform):
        raise TypeError(f"transform must be a callable, got {type(transform).__name__!r}")

    return SpruceUpConfig(sources=sources, target=target, embeddings=embeddings, transform=transform)
