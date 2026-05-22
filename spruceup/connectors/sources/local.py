import hashlib
import os
import pathlib
from dataclasses import dataclass

from ..base import SourceConnector
from ...hashing import hash_file_path


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
        from spruceup.monitoring.monitor import LocalFileWatcher
        return LocalFileWatcher(self.watched_dir, data_source_id, self.source_type)

    def display_name(self, identifier: str) -> str:
        return pathlib.Path(identifier).name

    def decode_content(self, raw_content: bytes) -> str:
        return raw_content.decode("utf-8", errors="replace")

    async def fetch(self, task):
        from spruceup.models import SpruceFile
        path = task.identifier
        with open(path, "rb") as f:
            raw_content = f.read()
        stat = os.stat(path)
        content_hash = hashlib.blake2b(raw_content, digest_size=16).digest()
        file_type = pathlib.Path(path).suffix.lstrip(".")
        return SpruceFile(
            file_id=hash_file_path(path),
            file_path=path,
            inode=stat.st_ino,
            mtime=stat.st_mtime,
            content_hash=content_hash,
            file_type=file_type,
            data_source_id=task.data_source_id,
            raw_content=raw_content,
            chunks=[],
        )
