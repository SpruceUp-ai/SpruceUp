import dataclasses
import hashlib
import inspect
import json
import typing
from typing import Callable

DIGEST_SIZE = 16


def hash_chunk_content(obj, vector_column: str) -> bytes:
    # Excludes the vector column: the vector is derived from the text, so
    # including it would make every re-embed look like a content change.
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        data = dataclasses.asdict(obj)
    else:
        data = obj.__dict__
    filtered = {k: v for k, v in data.items() if k != vector_column}
    serialized = json.dumps(filtered, sort_keys=True, default=str).encode()
    return hashlib.blake2b(serialized, digest_size=DIGEST_SIZE).digest()


def hash_transform(func: Callable) -> bytes:
    return hashlib.blake2b(
        inspect.getsource(func).encode(), digest_size=DIGEST_SIZE
    ).digest()


def hash_schema(schema: type, vector_column: str) -> str:
    hints = typing.get_type_hints(schema)
    parts = [f"{name}={tp!s}" for name, tp in sorted(hints.items())]
    parts.append(f"__vector_column__={vector_column}")
    return hashlib.blake2b(
        "|".join(parts).encode(), digest_size=DIGEST_SIZE
    ).hexdigest()


def hash_text(text: str) -> bytes:
    return hashlib.blake2b(text.encode(), digest_size=DIGEST_SIZE).digest()


def hash_args(
    fn: Callable, args: tuple, kwargs: dict, sig: inspect.Signature | None = None
) -> bytes:
    sig = sig if sig is not None else inspect.signature(fn)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    payload = json.dumps(_normalize(dict(bound.arguments)), sort_keys=True).encode()
    return hashlib.blake2b(payload, digest_size=DIGEST_SIZE).digest()


def _normalize(obj):
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_normalize(x) for x in obj]
    if isinstance(obj, float):
        return repr(obj)
    return obj
