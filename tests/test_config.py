import pytest

from spruceup.config import SpruceUpConfig, defineConfig
from spruceup.connectors.base import (
    EmbedderConnector,
    SourceConnector,
    TargetConnector,
)

class FakeSource(SourceConnector):
    @property
    def source_type(self) -> str:
        return "local"

    @property
    def source_identifier(self) -> str:
        return "/corpus"

    def create_watcher(self, data_source_id: int):
        return None

    @classmethod
    async def validate(cls, sources) -> None:
        pass

    def is_supported(self, file_identifier: str) -> bool:
        return True

    async def fetch(self, task):
        return None

    def display_name(self, identifier: str) -> str:
        return identifier

    def decode_content(self, raw_content: bytes) -> str:
        return raw_content.decode()


class FakeTarget(TargetConnector):
    @property
    def display_name(self) -> str:
        return "fake_target"

    def ensure_table_exists(self, embedding_dimensions: int) -> None:
        pass

    async def sync(self, upserts, deletes) -> None:
        pass


class FakeEmbedder(EmbedderConnector):
    async def embed_batch(self, batch):
        return [[0.0] for _ in batch]


async def a_transform(*, file_props, embed):
    return []


def valid_kwargs(**overrides):
    kwargs = dict(
        sources=[FakeSource()],
        target=FakeTarget(),
        embedder=FakeEmbedder(),
        transform=a_transform,
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_config_returns_populated_dataclass():
    src, tgt, emb = FakeSource(), FakeTarget(), FakeEmbedder()
    config = defineConfig(sources=[src], target=tgt, embedder=emb, transform=a_transform)

    assert isinstance(config, SpruceUpConfig)
    assert config.sources == [src]
    assert config.target is tgt
    assert config.embedder is emb
    assert config.transform is a_transform


# ---------------------------------------------------------------------------
# sources validation
# ---------------------------------------------------------------------------

def test_sources_not_a_list_raises():
    with pytest.raises(ValueError, match="non-empty list"):
        defineConfig(**valid_kwargs(sources=FakeSource()))


def test_empty_sources_raises():
    with pytest.raises(ValueError, match="non-empty list"):
        defineConfig(**valid_kwargs(sources=[]))


def test_source_element_wrong_type_raises():
    with pytest.raises(TypeError, match=r"sources\[1\] must be a SourceConnector"):
        defineConfig(**valid_kwargs(sources=[FakeSource(), object()]))


# ---------------------------------------------------------------------------
# target / embedder / transform validation
# ---------------------------------------------------------------------------

def test_target_wrong_type_raises():
    with pytest.raises(TypeError, match="target must be a TargetConnector"):
        defineConfig(**valid_kwargs(target=object()))


def test_embedder_wrong_type_raises():
    with pytest.raises(TypeError, match="embedder must be an EmbedderConnector"):
        defineConfig(**valid_kwargs(embedder=object()))


def test_non_callable_transform_raises():
    with pytest.raises(TypeError, match="transform must be a callable"):
        defineConfig(**valid_kwargs(transform="not callable"))
