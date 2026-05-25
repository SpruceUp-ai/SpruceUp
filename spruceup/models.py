from dataclasses import dataclass, field
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

@dataclass
class SyncTask:
    source_type: str          # "local", "google_drive", etc.
    identifier: str           # file path, Drive file ID, etc. (new path for moves)
    change_type: str          # "upsert" | "delete" | "move"
    old_identifier: str | None = field(default=None)  # previous path; only set for "move"
    data_source_id: int = field(default=0)
