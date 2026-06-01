import array
import dataclasses
import hashlib
import inspect
import json
import pathlib
from typing import Callable

DIGEST_SIZE = 16  # 16 bytes matches BINARY(16) in schema


def hash_source_ref(source_ref: str) -> bytes:
    return hashlib.blake2b(source_ref.encode(), digest_size=DIGEST_SIZE).digest()


def hash_chunk_id(file_path: str, ordinal: int) -> bytes:
    return hashlib.blake2b(f"{file_path}{ordinal}".encode(), digest_size=DIGEST_SIZE).digest()


def hash_object(obj) -> bytes:
    data = dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj.__dict__
    serialized = json.dumps(data, sort_keys=True, default=str).encode()
    return hashlib.blake2b(serialized, digest_size=DIGEST_SIZE).digest()


def hash_chunk_content(obj) -> bytes:
    data = dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj.__dict__
    filtered = {
        k: v for k, v in data.items()
        if not (isinstance(v, list) and v and isinstance(v[0], float))
    }
    serialized = json.dumps(filtered, sort_keys=True, default=str).encode()
    return hashlib.blake2b(serialized, digest_size=DIGEST_SIZE).digest()


def hash_chunk_text(text: str) -> bytes:
    """Hash only the embeddable text — NOT the surrounding chunk object.

    The embedding cache keys on this so that a metadata-only transform edit
    (which changes user_chunk_object_hash but leaves the embedded text alone)
    still hits the cache. Reusing hash_chunk_content here would defeat the
    feature's headline win.
    """
    return hashlib.blake2b(text.encode(), digest_size=DIGEST_SIZE).digest()


def pack_floats(values: list[float]) -> bytes:
    """Pack a vector as little-endian float32 bytes for BLOB storage.

    float32 is exact for the embedder's float32 API output; float64 would
    double the size for precision the source doesn't carry.
    """
    return array.array("f", values).tobytes()


def unpack_floats(blob: bytes) -> list[float]:
    """Inverse of pack_floats — decode float32 bytes back to a list."""
    arr = array.array("f")
    arr.frombytes(blob)
    return arr.tolist()


def hash_transform(func: Callable) -> bytes:
    return hashlib.blake2b(inspect.getsource(func).encode(), digest_size=DIGEST_SIZE).digest()


def hash_file_content(p: pathlib.Path) -> bytes:
    h = hashlib.blake2b(digest_size=DIGEST_SIZE)
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.digest()


def hash_args(fn: Callable, args: tuple, kwargs: dict) -> bytes:
    sig = inspect.signature(fn)
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
