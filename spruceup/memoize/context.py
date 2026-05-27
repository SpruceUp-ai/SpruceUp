from contextvars import ContextVar

_memo_manifest_var:  ContextVar = ContextVar('_memo_manifest',  default=None)
_memo_file_id_var:   ContextVar = ContextVar('_memo_file_id',   default=None)
_memo_temp_keys_var: ContextVar = ContextVar('_memo_temp_keys', default=None)
# Shared SQLite connection for all memoize reads/writes within one transform run.
# Avoids the open/close overhead that would otherwise happen on every memoized call.
_memo_conn_var:      ContextVar = ContextVar('_memo_conn',      default=None)
# [hits, total] counter reset per file; used to log a summary after transform.
_memo_stats_var:     ContextVar = ContextVar('_memo_stats',     default=None)
