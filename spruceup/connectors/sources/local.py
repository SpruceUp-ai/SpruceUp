import hashlib
import os
import pathlib
from dataclasses import dataclass

from ..base import SourceConnector
from ...utils.hashing import hash_source_ref


@dataclass
class LocalFilesSource(SourceConnector):
    watched_dir: str

    @property
    def source_type(self) -> str:
        return "local"

    @property
    def source_identifier(self) -> str:
        return str(pathlib.Path(self.watched_dir).resolve())

    def create_watcher(self, data_source_id: int):
        from spruceup.monitoring.local_file_watcher import LocalFileWatcher
        return LocalFileWatcher(self.watched_dir, data_source_id, self.source_type)

    def display_name(self, identifier: str) -> str:
        return pathlib.Path(identifier).name

    def decode_content(self, raw_content: bytes) -> str:
        return raw_content.decode("utf-8", errors="replace")

    async def fetch(self, task):
        from spruceup.models import SpruceFile
        path = task.identifier
        with open(path, "rb") as file:
            raw_content = file.read()
        file_stats = os.stat(path)
        content_hash = hashlib.blake2b(raw_content, digest_size=16).digest()
        file_type = pathlib.Path(path).suffix.lstrip(".")
        return SpruceFile(
            file_id=hash_source_ref(path),
            source_ref=path,
            content_hash=content_hash,
            file_type=file_type,
            data_source_id=task.data_source_id,
            raw_content=raw_content,
            chunks=[],
            source_metadata={
                "inode": file_stats.st_ino,
                "mtime": file_stats.st_mtime,
                "modified_at": file_stats.st_mtime,
            },
        )
