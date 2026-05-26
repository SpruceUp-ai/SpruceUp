from contextvars import ContextVar

_memo_manifest_var:  ContextVar = ContextVar('_memo_manifest',  default=None)
_memo_file_id_var:   ContextVar = ContextVar('_memo_file_id',   default=None)
_memo_temp_keys_var: ContextVar = ContextVar('_memo_temp_keys', default=None)
