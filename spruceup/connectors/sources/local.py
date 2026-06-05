import hashlib
import os
import pathlib
from dataclasses import dataclass

from ..base import SourceConnector, SUPPORTED_EXTENSIONS


@dataclass
class LocalFilesSource(SourceConnector):
    watched_dir: str

    @property
    def source_type(self) -> str:
        return "local"

    @property
    def source_identifier(self) -> str:
        return str(pathlib.Path(self.watched_dir).resolve())

    @classmethod
    async def validate(cls, sources: list["LocalFilesSource"]) -> None:
        resolved = [pathlib.Path(s.watched_dir).resolve() for s in sources]
        for i, path_a in enumerate(resolved):
            for path_b in resolved[i + 1:]:
                if path_b.is_relative_to(path_a) or path_a.is_relative_to(path_b):
                    ancestor, descendant = (
                        (path_a, path_b) if path_b.is_relative_to(path_a) else (path_b, path_a)
                    )
                    raise ValueError(
                        f"LocalFilesSource {str(ancestor)!r} is an ancestor of "
                        f"{str(descendant)!r}. Nested watched directories cause duplicate processing."
                    )

    def is_supported(self, file_identifier: str) -> bool:
        return pathlib.Path(file_identifier).suffix.lstrip(".").lower() in SUPPORTED_EXTENSIONS

    def create_watcher(self, data_source_id: int):
        from spruceup.monitoring.local_file_watcher import LocalFileWatcher
        return LocalFileWatcher(self.watched_dir, data_source_id, self.source_type, self.is_supported)

    def decode_content(self, raw_content: bytes) -> str:
        return raw_content.decode("utf-8", errors="replace")

    async def fetch(self, task, manifest):
        from spruceup.models import SpruceFile
        # file_id is "inode:path" — this source owns that format.
        path = task.current_file_id.split(":", 1)[1]
        file_stats = os.stat(path)
        file_id = f"{file_stats.st_ino}:{path}"
        file_type = pathlib.Path(path).suffix.lstrip(".")

        raw_content = None
        if task.use_manifest_cache:
            if manifest.get_file_modified_at(file_id) == file_stats.st_mtime:
                raw_content = manifest.get_raw_content(file_id)
        if raw_content is None:
            with open(path, "rb") as f:
                raw_content = f.read()

        content_hash = hashlib.blake2b(raw_content, digest_size=16).digest()
        return SpruceFile(
            file_id=file_id,
            display_name=pathlib.Path(path).name,
            content_hash=content_hash,
            file_type=file_type,
            data_source_id=task.data_source_id,
            raw_content=raw_content,
            chunks=[],
            modified_at=file_stats.st_mtime,
        )
