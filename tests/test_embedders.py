import pytest

from spruceup.connectors.base import EmbeddingConfigError, TokenExpiredError

from fakes import FakeEmbedder


class RaisingEmbedder(FakeEmbedder):
    """embed_batch always fails — stands in for a bad model/credential at startup."""

    async def embed_batch(self, batch):
        raise TokenExpiredError("rejected credentials")


class _FakeOutcome:
    """The only surface _refresh_token_before_retry reads off a tenacity outcome."""

    def __init__(self, exc):
        self._exc = exc

    def exception(self):
        return self._exc


class _FakeRetryState:
    """The only surface _refresh_token_before_retry reads off a RetryCallState."""

    def __init__(self, exc):
        self.outcome = _FakeOutcome(exc)


async def test_health_check_wraps_failure_as_config_error():
    embedder = RaisingEmbedder()
    with pytest.raises(EmbeddingConfigError):
        await embedder.health_check()


def test_refresh_invalidates_client_on_token_expiry_with_callable_key():
    embedder = FakeEmbedder(api_key=lambda: "fresh-token")
    embedder._client = object()

    embedder._refresh_token_before_retry(_FakeRetryState(TokenExpiredError("expired")))

    assert embedder._client is None
