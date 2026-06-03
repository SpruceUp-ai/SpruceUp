from contextvars import ContextVar

_embed_manifest_var:    ContextVar = ContextVar('_embed_manifest',    default=None)
_embed_file_id_var:     ContextVar = ContextVar('_embed_file_id',     default=None)
_embed_used_hashes_var: ContextVar = ContextVar('_embed_used_hashes', default=None)
_embed_stats_var:       ContextVar = ContextVar('_embed_stats',       default=None)
