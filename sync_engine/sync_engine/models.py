from dataclasses import dataclass
from typing import Any, Type


# This class doesn't make sense in our source code. But it's fine as a placeholder
# for testing. The user will be defining the class that plays this role.
@dataclass
class UserDefinedChunkSchema:
    """Base class users subclass to define their target table schema."""
    id: Any # this is non-sense, but again, doesn't matter since this won't hang around
    chunk_text: str
    chunk_embedding: list[float]


@dataclass
class ChunkWrapper:
    user_chunk: UserDefinedChunkSchema
    user_chunk_object_hash: bytes
    ordinal: int
    chunk_id: bytes


@dataclass
class File:
    file_id: bytes
    file_path: str
    mtime: float
    content_hash: bytes
    transform_hash: bytes
    file_type: str
    data_source_id: int
    chunks: list[ChunkWrapper]


@dataclass
class TargetTableConfig:
    db_name: str
    table_name: str
    schema_class: Type[UserDefinedChunkSchema]
    primary_key: str
