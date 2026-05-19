import dataclasses
import hashlib
import inspect
import json
import pathlib
from typing import Callable

DIGEST_SIZE = 16  # 16 bytes matches BINARY(16) in schema


def hash_file_path(file_path: str) -> bytes:
    return hashlib.blake2b(file_path.encode(), digest_size=DIGEST_SIZE).digest()


def hash_chunk_id(file_path: str, ordinal: int) -> bytes:
    return hashlib.blake2b(f"{file_path}{ordinal}".encode(), digest_size=DIGEST_SIZE).digest()


def hash_object(obj) -> bytes:
    data = dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj.__dict__
    serialized = json.dumps(data, sort_keys=True, default=str).encode()
    return hashlib.blake2b(serialized, digest_size=DIGEST_SIZE).digest()


def hash_transform(func: Callable) -> bytes:
    return hashlib.blake2b(inspect.getsource(func).encode(), digest_size=DIGEST_SIZE).digest()


def hash_file_content(p: pathlib.Path) -> bytes:
    h = hashlib.blake2b(digest_size=DIGEST_SIZE)
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.digest()
