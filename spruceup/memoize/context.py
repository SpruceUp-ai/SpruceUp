from contextvars import ContextVar

_memo_manifest_var:  ContextVar = ContextVar('_memo_manifest',  default=None)
_memo_file_id_var:   ContextVar = ContextVar('_memo_file_id',   default=None)
_memo_temp_keys_var: ContextVar = ContextVar('_memo_temp_keys', default=None)
# [hits, total] counter reset per file; used to log a summary after transform.
_memo_stats_var:     ContextVar = ContextVar('_memo_stats',     default=None)

# CachingEmbedder appends the text-hashes it computes into this list in embed-call order;
# the coordinator reads this list after the transform and writes those exact hashes into `chunks.text_hash`.
# One computation, two readers — the cache lookup and the column cannot diverge.
_embed_text_hashes_var: ContextVar = ContextVar('_embed_text_hashes', default=None)
