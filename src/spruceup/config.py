from dataclasses import dataclass
from typing import Callable

from .connectors.base import EmbedderConnector, SourceConnector, TargetConnector


@dataclass
class SpruceUpConfig:
    sources: list[SourceConnector]
    target: TargetConnector
    embedder: EmbedderConnector
    transform: Callable
    cache_files: bool = False


def defineConfig(*, sources, target, embedder, transform, cache_files=False) -> SpruceUpConfig:
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list of source connectors")
    for i, source in enumerate(sources):
        if not isinstance(source, SourceConnector):
            raise TypeError(
                f"sources[{i}] must be a SourceConnector, got {type(source).__name__!r}"
            )

    if not isinstance(target, TargetConnector):
        raise TypeError(f"target must be a TargetConnector, got {type(target).__name__!r}")

    if not isinstance(embedder, EmbedderConnector):
        raise TypeError(
            f"embedder must be an EmbedderConnector, got {type(embedder).__name__!r}"
        )

    if not callable(transform):
        raise TypeError(f"transform must be a callable, got {type(transform).__name__!r}")

    if not isinstance(cache_files, bool):
        raise TypeError(f"cache_files must be a bool, got {type(cache_files).__name__!r}")

    return SpruceUpConfig(
        sources=sources,
        target=target,
        embedder=embedder,
        transform=transform,
        cache_files=cache_files,
    )
