import pytest

from spruceup.memoize.decorator import memoize
from spruceup.transform_context import TransformContext, transform_scope

from fakes import make_file


def _seed_file(manifest, file_id="file-1"):
    source_id = manifest.register_source("local", "src")
    manifest.upsert_file_row(make_file(file_id=file_id, data_source_id=source_id))


async def test_miss_then_hit_skips_recompute(manifest):
    _seed_file(manifest)
    calls = []

    @memoize(return_type=str)
    async def summarize(text: str) -> str:
        calls.append(text)
        return text.upper()

    ctx = TransformContext(manifest=manifest, file_id="file-1")
    with transform_scope(ctx):
        first = await summarize("hello")
        second = await summarize("hello")

    assert first == "HELLO"
    assert second == "HELLO"
    assert calls == ["hello"]
    assert ctx.memo_total == 2
    assert ctx.memo_hits == 1
    assert len(ctx.used_memoized_subfn_call_keys) == 1


async def test_distinct_args_each_miss(manifest):
    _seed_file(manifest)
    calls = []

    @memoize(return_type=str)
    async def summarize(text: str) -> str:
        calls.append(text)
        return text.upper()

    ctx = TransformContext(manifest=manifest, file_id="file-1")
    with transform_scope(ctx):
        await summarize("a")
        await summarize("b")

    assert calls == ["a", "b"]
    assert ctx.memo_total == 2
    assert ctx.memo_hits == 0
    assert len(ctx.used_memoized_subfn_call_keys) == 2


def test_sync_function_rejected_at_decoration():
    with pytest.raises(TypeError):

        @memoize(return_type=str)
        def summarize(text: str) -> str:
            return text.upper()


def test_unsupported_return_type_rejected():
    with pytest.raises(TypeError):
        memoize(return_type=set)


async def test_call_outside_transform_context_raises():
    @memoize(return_type=str)
    async def summarize(text: str) -> str:
        return text.upper()

    with pytest.raises(RuntimeError):
        await summarize("hello")
