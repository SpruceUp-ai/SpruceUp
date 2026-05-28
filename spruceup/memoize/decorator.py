import asyncio
import functools
import inspect

from ..utils.hashing import hash_transform, hash_args
from .context import _memo_manifest_var, _memo_file_id_var, _memo_temp_keys_var, _memo_conn_var, _memo_stats_var
from .serialization import validate_return_type, serialize, deserialize


def memoize(*, returns):
    """Cache subfunction results in SQLite, scoped per file.

    Supported return types: str, int, float, bool, list, dict.

    Known limitation: fn_hash covers only this function's own source. Changes
    to helper functions it calls will not invalidate the cache.

    Only valid when called from within the transform function passed to defineConfig().
    """
    validate_return_type(returns)

    def decorator(fn):
        fn_hash = hash_transform(fn)

        def _lookup(args, kwargs):
            manifest  = _memo_manifest_var.get()
            file_id   = _memo_file_id_var.get()
            temp_keys = _memo_temp_keys_var.get()
            conn      = _memo_conn_var.get()
            if manifest is None:
                raise RuntimeError(
                    f"@memoize function '{fn.__name__}' was called outside a transform "
                    "context. @memoize subfunctions may only be called from within the "
                    "transform function passed to defineConfig()."
                )
            args_h = hash_args(fn, args, kwargs)
            temp_keys.add((fn_hash, args_h))
            cached = manifest.get_memoized(file_id, fn_hash, args_h, conn)
            return manifest, file_id, args_h, cached, conn

        def _store(manifest, file_id, args_h, result, conn):
            manifest.set_memoized(file_id, fn_hash, args_h, serialize(result, returns), conn)

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                manifest, file_id, args_h, cached, conn = _lookup(args, kwargs)
                stats = _memo_stats_var.get()
                if stats is not None:
                    stats[1] += 1
                if cached is not None:
                    if stats is not None:
                        stats[0] += 1
                    return deserialize(cached, returns)
                if conn is not None:
                    conn.commit()
                result = await fn(*args, **kwargs)
                _store(manifest, file_id, args_h, result, conn)
                return result
        else:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                manifest, file_id, args_h, cached, conn = _lookup(args, kwargs)
                stats = _memo_stats_var.get()
                if stats is not None:
                    stats[1] += 1
                if cached is not None:
                    if stats is not None:
                        stats[0] += 1
                    return deserialize(cached, returns)
                result = fn(*args, **kwargs)
                _store(manifest, file_id, args_h, result, conn)
                return result

        return wrapper
    return decorator
