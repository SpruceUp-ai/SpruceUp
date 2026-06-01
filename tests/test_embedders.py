import math
import types

import pytest

from spruceup.connectors.embedders.cohere import CohereEmbedder
from spruceup.connectors.embedders.gemini import GeminiEmbedder, _l2_normalize
from spruceup.connectors.embedders.openai import OpenAIEmbedder
from spruceup.connectors.embedders.voyageai import VoyageAIEmbedder


# ---------------------------------------------------------------------------
# Fake async API clients (only the surface each embed_batch touches)
# ---------------------------------------------------------------------------

class _FakeOpenAIClient:
    def __init__(self, vectors):
        self.calls: list[dict] = []
        outer = self

        class _Embeddings:
            async def create(self, **kwargs):
                outer.calls.append(kwargs)
                return types.SimpleNamespace(
                    data=[types.SimpleNamespace(embedding=v) for v in vectors]
                )

        self.embeddings = _Embeddings()


class _FakeCohereClient:
    def __init__(self, vectors):
        self.calls: list[dict] = []

    async def embed(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(
            embeddings=types.SimpleNamespace(float_=[[9.0], [9.0]])
        )


class _FakeVoyageClient:
    def __init__(self, vectors):
        self.calls: list[dict] = []
        self._vectors = vectors

    async def embed(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(embeddings=self._vectors)


class _FakeGeminiClient:
    def __init__(self, vectors):
        self.calls: list[dict] = []
        outer = self

        class _Models:
            async def embed_content(self, **kwargs):
                outer.calls.append(kwargs)
                return types.SimpleNamespace(
                    embeddings=[types.SimpleNamespace(values=v) for v in vectors]
                )

        self.aio = types.SimpleNamespace(models=_Models())


# ===========================================================================
# OpenAI
# ===========================================================================

class TestOpenAIEmbedder:

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="requires an api_key"):
            OpenAIEmbedder(api_key="")

    def test_default_dimensions_from_known_model(self):
        emb = OpenAIEmbedder(api_key="k", model="text-embedding-3-small")
        assert emb.embedding_dimensions == 1536
        assert emb._dimensions_overridden is False

    def test_unknown_model_without_dimensions_raises(self):
        with pytest.raises(ValueError, match="unknown model"):
            OpenAIEmbedder(api_key="k", model="mystery-model")

    def test_explicit_dimensions_override(self):
        emb = OpenAIEmbedder(api_key="k", model="mystery-model", embedding_dimensions=42)
        assert emb.embedding_dimensions == 42
        assert emb._dimensions_overridden is True

    def test_get_client_is_lazy_and_cached(self):
        emb = OpenAIEmbedder(api_key="k")
        assert emb._client is None
        client = emb._get_client()
        assert client is not None
        assert emb._get_client() is client

    @pytest.mark.asyncio
    async def test_embed_batch_returns_vectors(self):
        emb = OpenAIEmbedder(api_key="k")
        emb._client = _FakeOpenAIClient([[1.0, 2.0], [3.0, 4.0]])
        result = await emb.embed_batch(["a", "b"])
        assert result == [[1.0, 2.0], [3.0, 4.0]]
        assert "dimensions" not in emb._client.calls[0]

    @pytest.mark.asyncio
    async def test_embed_batch_sends_dimensions_when_overridden(self):
        emb = OpenAIEmbedder(api_key="k", model="text-embedding-3-small", embedding_dimensions=256)
        emb._client = _FakeOpenAIClient([[0.0]])
        await emb.embed_batch(["a"])
        assert emb._client.calls[0]["dimensions"] == 256


# ===========================================================================
# Cohere
# ===========================================================================

class TestCohereEmbedder:

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="requires an api_key"):
            CohereEmbedder(api_key="")

    def test_default_dimensions_from_known_model(self):
        emb = CohereEmbedder(api_key="k", model="embed-english-v3.0")
        assert emb.embedding_dimensions == 1024

    def test_unknown_model_without_dimensions_raises(self):
        with pytest.raises(ValueError, match="unknown model"):
            CohereEmbedder(api_key="k", model="mystery-model")

    def test_embed_v4_rejects_disallowed_dimension(self):
        with pytest.raises(ValueError, match="only supports"):
            CohereEmbedder(api_key="k", model="embed-v4.0", embedding_dimensions=999)

    def test_embed_v4_accepts_allowed_dimension(self):
        emb = CohereEmbedder(api_key="k", model="embed-v4.0", embedding_dimensions=512)
        assert emb.embedding_dimensions == 512
        assert emb._dimensions_overridden is True

    def test_get_client_is_lazy_and_cached(self):
        emb = CohereEmbedder(api_key="k")
        assert emb._client is None
        client = emb._get_client()
        assert emb._get_client() is client

    @pytest.mark.asyncio
    async def test_embed_batch_returns_float_embeddings(self):
        emb = CohereEmbedder(api_key="k", model="embed-v4.0", embedding_dimensions=1024)
        emb._client = _FakeCohereClient(None)
        result = await emb.embed_batch(["a", "b"])
        assert result == [[9.0], [9.0]]
        assert emb._client.calls[0]["output_dimension"] == 1024


# ===========================================================================
# VoyageAI
# ===========================================================================

class TestVoyageAIEmbedder:

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="requires an api_key"):
            VoyageAIEmbedder(api_key="")

    def test_default_dimensions_from_known_model(self):
        emb = VoyageAIEmbedder(api_key="k", model="voyage-3-lite")
        assert emb.embedding_dimensions == 512

    def test_unknown_model_without_dimensions_raises(self):
        with pytest.raises(ValueError, match="unknown model"):
            VoyageAIEmbedder(api_key="k", model="mystery-model")

    def test_voyage_4_rejects_disallowed_dimension(self):
        with pytest.raises(ValueError, match="only supports"):
            VoyageAIEmbedder(api_key="k", model="voyage-4-large", embedding_dimensions=999)

    def test_voyage_4_accepts_allowed_dimension(self):
        emb = VoyageAIEmbedder(api_key="k", model="voyage-4-large", embedding_dimensions=2048)
        assert emb.embedding_dimensions == 2048

    def test_get_client_is_lazy_and_cached(self):
        emb = VoyageAIEmbedder(api_key="k")
        assert emb._client is None
        client = emb._get_client()
        assert emb._get_client() is client

    @pytest.mark.asyncio
    async def test_embed_batch_returns_vectors_and_passes_dimensions(self):
        emb = VoyageAIEmbedder(api_key="k", model="voyage-3", embedding_dimensions=1024)
        emb._client = _FakeVoyageClient([[1.0], [2.0]])
        result = await emb.embed_batch(["a", "b"])
        assert result == [[1.0], [2.0]]
        assert emb._client.calls[0]["output_dimension"] == 1024


# ===========================================================================
# Gemini
# ===========================================================================

class TestGeminiEmbedder:

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="requires an api_key"):
            GeminiEmbedder(api_key="")

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="unknown model"):
            GeminiEmbedder(api_key="k", model="mystery-model")

    def test_max_batch_size_over_limit_raises(self):
        with pytest.raises(ValueError, match="cannot exceed 100"):
            GeminiEmbedder(api_key="k", max_batch_size=101)

    def test_default_dimensions_and_no_normalization(self):
        emb = GeminiEmbedder(api_key="k", model="text-embedding-004")
        assert emb.embedding_dimensions == 768
        assert emb._needs_normalization is False

    def test_overridden_dimensions_flag_normalization(self):
        emb = GeminiEmbedder(api_key="k", model="gemini-embedding-001", embedding_dimensions=1536)
        assert emb.embedding_dimensions == 1536
        assert emb._dimensions_overridden is True
        assert emb._needs_normalization is True

    def test_get_client_is_lazy_and_cached(self):
        emb = GeminiEmbedder(api_key="k")
        assert emb._client is None
        client = emb._get_client()
        assert emb._get_client() is client

    @pytest.mark.asyncio
    async def test_embed_batch_returns_raw_vectors_without_normalization(self):
        emb = GeminiEmbedder(api_key="k", model="text-embedding-004")
        emb._client = _FakeGeminiClient([[3.0, 4.0]])
        result = await emb.embed_batch(["a"])
        assert result == [[3.0, 4.0]]

    @pytest.mark.asyncio
    async def test_embed_batch_normalizes_when_dimensions_overridden(self):
        emb = GeminiEmbedder(api_key="k", model="gemini-embedding-001", embedding_dimensions=1536)
        emb._client = _FakeGeminiClient([[3.0, 4.0]])
        result = await emb.embed_batch(["a"])
        assert result[0] == pytest.approx([0.6, 0.8])


class TestL2Normalize:

    def test_unit_normalizes_vector(self):
        assert _l2_normalize([3.0, 4.0]) == pytest.approx([0.6, 0.8])
        assert math.isclose(math.sqrt(sum(x * x for x in _l2_normalize([3.0, 4.0]))), 1.0)

    def test_zero_vector_returned_unchanged(self):
        assert _l2_normalize([0.0, 0.0]) == [0.0, 0.0]
