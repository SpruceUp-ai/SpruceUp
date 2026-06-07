from dataclasses import dataclass, field
from typing import Any


@dataclass
class FileProps:
    raw_content: str | bytes
    display_name: str
    file_type: str


@dataclass
class ChunkWrapper:
    user_chunk: Any
    user_chunk_object_hash: bytes


@dataclass
class SpruceFile:
    file_id: str
    display_name: str
    file_type: str
    data_source_id: int
    raw_content: str | bytes
    chunks: list[ChunkWrapper]
    modified_at: float
    force_upsert: bool = False

@dataclass
class SyncTask:
    source_type: str          # "local", "google_drive", etc.
    change_type: str          # "upsert" | "delete"
    current_file_id: str | None = field(default=None)
    data_source_id: int = field(default=0)
    use_manifest_cache: bool = field(default=False)
