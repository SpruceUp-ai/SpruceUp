from dataclasses import dataclass
from typing import Any, Type


@dataclass
class UserDefinedChunkSchema:
    """Optional base class for user-defined chunk schema dataclasses.

    Users may subclass this or define a standalone dataclass. Either way, the
    dataclass must follow these conventions:
      - chunk_text: str       — the text stored and used for embedding
      - chunk_embedding: list[float] — populated by the framework after embedding
      - at least one primary-key field matching the PRIMARY_KEY constant in the
        user's spruceup_pipeline.py
    """
    id: Any
    chunk_text: str
    chunk_embedding: list[float]


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
    transform_hash: bytes
    file_type: str
    data_source_id: int
    raw_content: str | bytes
    parsed_content: str | None
    chunk_strs: list[str]
    chunks: list[ChunkWrapper]


@dataclass
class TargetTableConfig:
    db_name: str
    table_name: str
    schema_class: type
    primary_key: str
