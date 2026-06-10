from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class TransformContext:
    manifest: object
    file_id: str
    used_memoized_subfn_call_keys: set = field(default_factory=set)
    memo_hits: int = 0
    memo_total: int = 0
    used_chunk_embedding_cache_keys: set = field(default_factory=set)
    embed_hits: int = 0
    embed_total: int = 0


_transform_context: ContextVar = ContextVar("_transform_context", default=None)


def get_transform_context() -> TransformContext | None:
    return _transform_context.get()


@contextmanager
def transform_scope(ctx: TransformContext):
    token = _transform_context.set(ctx)
    try:
        yield ctx
    finally:
        _transform_context.reset(token)
