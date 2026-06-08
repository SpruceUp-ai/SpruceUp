from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class TransformContext:
    manifest: object
    file_id: str
    memo_temp_keys: set = field(default_factory=set)
    memo_hits: int = 0
    memo_total: int = 0
    embed_used_hashes: set = field(default_factory=set)
    embed_hits: int = 0
    embed_total: int = 0


_transform_context: ContextVar = ContextVar("_transform_context", default=None)
