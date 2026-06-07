import dataclasses
import hashlib
import inspect
import json
import pathlib
import typing
from typing import Callable

DIGEST_SIZE = 16  # 16 bytes matches BINARY(16) in schema


def hash_chunk_content(obj) -> bytes:
    data = dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj.__dict__
    filtered = {
        k: v for k, v in data.items()
        if not (isinstance(v, list) and v and isinstance(v[0], float))
    }
    serialized = json.dumps(filtered, sort_keys=True, default=str).encode()
    return hashlib.blake2b(serialized, digest_size=DIGEST_SIZE).digest()


def hash_transform(func: Callable) -> bytes:
    return hashlib.blake2b(inspect.getsource(func).encode(), digest_size=DIGEST_SIZE).digest()


def hash_schema(schema: type, vector_column: str) -> str:
    """Stable fingerprint of a schema's columns + designated vector column.

    Changes to field names, field types, or the vector column flip this hash,
    which drives a drop+recreate of the target table.
    """
    hints = typing.get_type_hints(schema)
    parts = [f"{name}={tp!s}" for name, tp in sorted(hints.items())]
    parts.append(f"__vector_column__={vector_column}")
    return hashlib.blake2b("|".join(parts).encode(), digest_size=DIGEST_SIZE).hexdigest()


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
