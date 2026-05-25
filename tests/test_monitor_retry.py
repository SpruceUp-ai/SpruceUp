# tests/test_monitor_retry.py
import pytest
from spruceup.monitoring.monitor import _with_retry


@pytest.mark.asyncio
async def test_retry_recovers_from_transient_failure():
    calls = {"n": 0}
    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
    await _with_retry(flaky, max_attempts=5)
    assert calls["n"] == 3   # failed twice, succeeded on third


@pytest.mark.asyncio
async def test_retry_reraises_after_ceiling():
    calls = {"n": 0}
    async def broken():
        calls["n"] += 1
        raise RuntimeError("permanent")
    with pytest.raises(RuntimeError, match="permanent"):
        await _with_retry(broken, max_attempts=3)
    assert calls["n"] == 3   # exactly ceiling attempts, no more


@pytest.mark.asyncio
async def test_retry_success_runs_once():
    calls = {"n": 0}
    async def ok():
        calls["n"] += 1
    await _with_retry(ok)
    assert calls["n"] == 1   # no spurious retries
