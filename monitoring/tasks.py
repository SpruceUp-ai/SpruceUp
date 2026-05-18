from dataclasses import dataclass


@dataclass
class SyncTask:
    source_type: str  # "local", "google_drive", etc.
    identifier: str   # file path, Drive file ID, etc.
    change_type: str  # "upsert" | "delete"
