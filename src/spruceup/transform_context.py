from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class TransformContext:
    """Per-file state shared with @memoize subfunctions and the embedding cache.

    Set once by the Coordinator before a file's transform runs, then read back
    by the @memoize decorator and the EmbeddingBatcher, which execute inside it.
    It is a single mutable object propagated through one ContextVar — contextvars
    flow into awaited coroutines within the same task — so the hit/total counters
    and used-key sets those consumers mutate are visible to the Coordinator after
    the transform completes (for cache sweeping and the per-file log lines).
    """

    manifest: object
    file_id: str
    memo_temp_keys: set = field(default_factory=set)
    memo_hits: int = 0
    memo_total: int = 0
    embed_used_hashes: set = field(default_factory=set)
    embed_hits: int = 0
    embed_total: int = 0


_transform_context: ContextVar = ContextVar("_transform_context", default=None)
