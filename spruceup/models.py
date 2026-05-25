from dataclasses import dataclass
from typing import Any


@dataclass
class ChunkWrapper:
    user_chunk: Any
    user_chunk_object_hash: bytes
    ordinal: int
    chunk_id: bytes


@dataclass
class SpruceFile:
    file_id: bytes
    file_path: str
    inode: int
    mtime: float
    content_hash: bytes
    file_type: str
    data_source_id: int
    raw_content: str | bytes
    chunks: list[ChunkWrapper]
