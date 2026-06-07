import functools
import inspect

from ..utils.hashing import hash_args, hash_transform
from ..transform_context import _transform_context
from .serialization import validate_return_type, serialize, deserialize

_memoize_fn_hashes: set[bytes] = set()


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
        _sig = inspect.signature(fn)
        _memoize_fn_hashes.add(fn_hash)

        def _lookup(args, kwargs):
            ctx = _transform_context.get()
            if ctx is None:
                raise RuntimeError(
                    f"@memoize function '{fn.__name__}' was called outside a transform "
                    "context. @memoize subfunctions may only be called from within the "
                    "transform function passed to defineConfig()."
                )
            args_h = hash_args(fn, args, kwargs, sig=_sig)
            ctx.memo_temp_keys.add((fn_hash, args_h))
            cached = ctx.manifest.get_memoized(ctx.file_id, fn_hash, args_h)
            return ctx, args_h, cached

        def _store(ctx, args_h, result):
            ctx.manifest.set_memoized(ctx.file_id, fn_hash, args_h, serialize(result, returns))

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                ctx, args_h, cached = _lookup(args, kwargs)
                ctx.memo_total += 1
                if cached is not None:
                    ctx.memo_hits += 1
                    return deserialize(cached, returns)
                result = await fn(*args, **kwargs)
                _store(ctx, args_h, result)
                return result
        else:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                ctx, args_h, cached = _lookup(args, kwargs)
                ctx.memo_total += 1
                if cached is not None:
                    ctx.memo_hits += 1
                    return deserialize(cached, returns)
                result = fn(*args, **kwargs)
                _store(ctx, args_h, result)
                return result

        return wrapper
    return decorator
