from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FileProps:
    raw_content: str
    source_ref: str
    display_name: str
    file_type: str
    modified_at: Optional[float]


@dataclass
class ChunkWrapper:
    user_chunk: Any
    user_chunk_object_hash: bytes
    ordinal: int
    chunk_id: bytes
    # Hash of the embeddable text (ADR-0001 provenance). Sourced from the
    # CachingEmbedder ContextVar, not re-derived from user_chunk. None when the
    # cache is disabled or the transform made no embed() call for this chunk.
    text_hash: bytes | None = None


@dataclass
class SpruceFile:
    file_id: bytes
    source_ref: str
    display_name: str
    content_hash: bytes
    file_type: str
    data_source_id: int
    raw_content: str | bytes
    chunks: list[ChunkWrapper]
    source_metadata: dict = field(default_factory=dict)

@dataclass
class SyncTask:
    source_type: str          # "local", "google_drive", etc.
    identifier: str           # file path, Drive file ID, etc. (new path for moves)
    change_type: str          # "upsert" | "delete" | "move"
    old_identifier: str | None = field(default=None)  # previous path; only set for "move"
    data_source_id: int = field(default=0)
