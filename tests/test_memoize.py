"""
Tests for the @memoize decorator.

SQLite manifest runs against a real temp file.
All memoized functions defined inline per test so fn_hash is unique per test.
"""

import pytest

from spruceup.manifest import Manifest
from spruceup.memoize import memoize
from spruceup.memoize.context import (
    _memo_file_id_var,
    _memo_manifest_var,
    _memo_temp_keys_var,
)
from spruceup.utils.hashing import hash_source_ref

FILE_PATH = "corpus/test_doc.txt"
FILE_ID = hash_source_ref(FILE_PATH)

FN_HASH = b"\x01" * 16
ARGS_HASH_A = b"\x02" * 16
ARGS_HASH_B = b"\x03" * 16


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_manifest(tmp_path):
    return str(tmp_path / "manifest.db")


@pytest.fixture
def manifest(tmp_manifest):
    return Manifest(tmp_manifest)


@pytest.fixture
def memo_ctx(manifest):
    """Set all three ContextVars for FILE_ID and return the temp_keys set."""
    temp_keys: set[tuple[bytes, bytes]] = set()
    _memo_manifest_var.set(manifest)
    _memo_file_id_var.set(FILE_ID)
    _memo_temp_keys_var.set(temp_keys)
    return temp_keys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def seed_file_row(manifest, file_id=FILE_ID, file_path=FILE_PATH):
    manifest._conn.execute(
        "INSERT OR IGNORE INTO files (id, source_ref) VALUES (?, ?)",
        (file_id, file_path),
    )


def cache_row_count(manifest, file_id=FILE_ID) -> int:
    return manifest._conn.execute(
        "SELECT COUNT(*) FROM memoize_cache WHERE file_id=?", (file_id,)
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# 1 & 2 — Cache miss and cache hit (sync function)
# ---------------------------------------------------------------------------


def test_cache_miss_calls_function_and_stores_result(manifest, memo_ctx):
    seed_file_row(manifest)
    call_count = 0

    @memoize(returns=str)
    def fn(text: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"result:{text}"

    result = fn("hello")
    assert result == "result:hello"
    assert call_count == 1
    assert cache_row_count(manifest) == 1


def test_cache_hit_does_not_call_function_again(manifest, memo_ctx):
    seed_file_row(manifest)
    call_count = 0

    @memoize(returns=str)
    def fn(text: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"result:{text}"

    result1 = fn("hello")
    result2 = fn("hello")
    assert result1 == result2 == "result:hello"
    assert call_count == 1
    assert cache_row_count(manifest) == 1


# ---------------------------------------------------------------------------
# Async function support
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cache_hit_async_function(manifest, memo_ctx):
    seed_file_row(manifest)
    call_count = 0

    @memoize(returns=str)
    async def fn(text: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"async:{text}"

    result1 = await fn("hello")
    result2 = await fn("hello")
    assert result1 == result2 == "async:hello"
    assert call_count == 1


# ---------------------------------------------------------------------------
# 3 & 4 — Sweep: stale removed, live preserved
# ---------------------------------------------------------------------------


def test_sweep_removes_stale_entries(manifest):
    seed_file_row(manifest)
    manifest.set_memoized(FILE_ID, FN_HASH, ARGS_HASH_A, b'"value_a"')
    manifest.set_memoized(FILE_ID, FN_HASH, ARGS_HASH_B, b'"value_b"')
    assert cache_row_count(manifest) == 2

    manifest.sweep_memoized(FILE_ID, {(FN_HASH, ARGS_HASH_A)})

    assert manifest.get_memoized(FILE_ID, FN_HASH, ARGS_HASH_A) is not None
    assert manifest.get_memoized(FILE_ID, FN_HASH, ARGS_HASH_B) is None


def test_sweep_preserves_live_entry(manifest):
    seed_file_row(manifest)
    manifest.set_memoized(FILE_ID, FN_HASH, ARGS_HASH_A, b'"value_a"')

    manifest.sweep_memoized(FILE_ID, {(FN_HASH, ARGS_HASH_A)})

    assert cache_row_count(manifest) == 1


def test_sweep_with_empty_temp_keys_clears_all(manifest):
    seed_file_row(manifest)
    manifest.set_memoized(FILE_ID, FN_HASH, ARGS_HASH_A, b'"value_a"')
    manifest.set_memoized(FILE_ID, FN_HASH, ARGS_HASH_B, b'"value_b"')

    manifest.sweep_memoized(FILE_ID, set())

    assert cache_row_count(manifest) == 0


def test_sweep_does_not_leak_keys_across_files(manifest):
    # The manifest reuses one long-lived connection, so the sweep's TEMP table
    # persists between calls. Each sweep must start from a clean key set, or a
    # later file would wrongly preserve a stale entry whose (fn, args) signature
    # happened to be swept-as-live for an earlier file.
    file_a = hash_source_ref("corpus/a.txt")
    file_b = hash_source_ref("corpus/b.txt")
    seed_file_row(manifest, file_id=file_a, file_path="corpus/a.txt")
    seed_file_row(manifest, file_id=file_b, file_path="corpus/b.txt")

    manifest.set_memoized(file_a, FN_HASH, ARGS_HASH_A, b'"a_live"')
    manifest.sweep_memoized(file_a, {(FN_HASH, ARGS_HASH_A)})

    # File B has a stale entry with the same signature A just kept alive.
    manifest.set_memoized(file_b, FN_HASH, ARGS_HASH_A, b'"b_stale"')
    manifest.set_memoized(file_b, FN_HASH, ARGS_HASH_B, b'"b_live"')
    manifest.sweep_memoized(file_b, {(FN_HASH, ARGS_HASH_B)})

    assert manifest.get_memoized(file_b, FN_HASH, ARGS_HASH_A) is None
    assert manifest.get_memoized(file_b, FN_HASH, ARGS_HASH_B) is not None


# ---------------------------------------------------------------------------
# 5 — Move: cache reassigned to new file_id
# ---------------------------------------------------------------------------


def test_move_preserves_cache_under_same_file_id(manifest):
    # file_id is stable across renames (inode-based for local, Drive ID for Drive).
    # move_file only updates source_ref; the memoize_cache FK is on file_id so the
    # cache survives the rename without any special migration.
    old_path = "corpus/old.txt"
    new_path = "corpus/new.txt"
    file_id = hash_source_ref(old_path)

    seed_file_row(manifest, file_id=file_id, file_path=old_path)
    manifest.set_memoized(file_id, FN_HASH, ARGS_HASH_A, b'"cached"')

    manifest.update_file_ref(file_id, new_path)

    assert manifest.get_memoized(file_id, FN_HASH, ARGS_HASH_A) is not None


# ---------------------------------------------------------------------------
# 6 — Delete cascade
# ---------------------------------------------------------------------------


def test_delete_file_row_cascades_to_memoize_cache(manifest):
    seed_file_row(manifest)
    manifest.set_memoized(FILE_ID, FN_HASH, ARGS_HASH_A, b'"value"')
    assert cache_row_count(manifest) == 1

    manifest.delete_file_row(FILE_ID)

    assert cache_row_count(manifest) == 0


# ---------------------------------------------------------------------------
# 7 — RuntimeError outside transform context
# ---------------------------------------------------------------------------


def test_memoize_raises_outside_transform_context():
    _memo_manifest_var.set(None)

    @memoize(returns=str)
    def fn(text: str) -> str:
        return text

    with pytest.raises(RuntimeError, match="outside a transform context"):
        fn("hello")


# ---------------------------------------------------------------------------
# 8 — Type round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "returns, value",
    [
        (str, "hello"),
        (int, 42),
        (float, 3.14),
        (bool, True),
        (list, [1, "two", 3.0]),
        (dict, {"key": "value", "num": 42}),
    ],
)
def test_type_round_trips(manifest, memo_ctx, returns, value):
    seed_file_row(manifest)

    @memoize(returns=returns)
    def fn(x):
        return value

    result1 = fn("arg")
    result2 = fn("arg")  # cache hit
    assert result1 == value
    assert result2 == value
    assert type(result2) is type(value)


# ---------------------------------------------------------------------------
# 9 — dict with non-string keys raises at call time
# ---------------------------------------------------------------------------


def test_dict_non_string_keys_raises(manifest, memo_ctx):
    seed_file_row(manifest)

    @memoize(returns=dict)
    def fn(x):
        return {1: "value"}

    with pytest.raises(TypeError, match="string keys"):
        fn("arg")


# ---------------------------------------------------------------------------
# Decoration-time errors
# ---------------------------------------------------------------------------


def test_unsupported_return_type_raises_at_decoration_time():
    class CustomType:
        pass

    with pytest.raises(TypeError, match="not supported"):

        @memoize(returns=CustomType)
        def fn(x):
            return CustomType()
