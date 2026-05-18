import dataclasses
import hashlib
import json

DIGEST_SIZE = 16  # 16 bytes matches BINARY(16) in schema


def hash_file_path(file_path: str) -> bytes:
    return hashlib.blake2b(file_path.encode(), digest_size=DIGEST_SIZE).digest()


def hash_chunk_id(file_path: str, ordinal: int) -> bytes:
    return hashlib.blake2b(f"{file_path}{ordinal}".encode(), digest_size=DIGEST_SIZE).digest()


def hash_object(obj) -> bytes:
    data = dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj.__dict__
    serialized = json.dumps(data, sort_keys=True, default=str).encode()
    return hashlib.blake2b(serialized, digest_size=DIGEST_SIZE).digest()
