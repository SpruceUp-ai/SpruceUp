import dataclasses
import hashlib
import inspect
import json
import pathlib
from typing import Callable

DIGEST_SIZE = 16  # 16 bytes matches BINARY(16) in schema


def hash_source_ref(source_ref: str) -> bytes:
    return hashlib.blake2b(source_ref.encode(), digest_size=DIGEST_SIZE).digest()


def hash_inode(inode: int) -> bytes:
    return hashlib.blake2b(inode.to_bytes(8, "little"), digest_size=DIGEST_SIZE).digest()



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
