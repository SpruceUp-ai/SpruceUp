from dataclasses import dataclass, field


@dataclass
class SyncTask:
    source_type: str          # "local", "google_drive", etc.
    identifier: str           # file path, Drive file ID, etc. (new path for moves)
    change_type: str          # "upsert" | "delete" | "move"
    old_identifier: str | None = field(default=None)  # previous path; only set for "move"
    data_source_id: int = field(default=0)
