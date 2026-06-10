import os
import pathlib
from dataclasses import dataclass
from typing import cast

from ..base import SourceConnector, SUPPORTED_EXTENSIONS


def make_file_id(inode: int, path: str) -> str:
    return f"{inode}:{path}"


def file_id_to_inode(file_id: str) -> int:
    return int(file_id.split(":", 1)[0])


def file_id_to_path(file_id: str) -> str:
    return file_id.split(":", 1)[1]


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
    async def validate(cls, sources: list["SourceConnector"]) -> None:
        typed = cast("list[LocalFilesSource]", sources)
        resolved = [pathlib.Path(s.watched_dir).resolve() for s in typed]
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
        return LocalFileWatcher(self.watched_dir, data_source_id, self.is_supported)

    async def fetch(self, task, manifest):
        from spruceup.models import SpruceFile
        path = file_id_to_path(task.current_file_id)
        file_stats = os.stat(path)
        file_id = make_file_id(file_stats.st_ino, path)
        file_type = pathlib.Path(path).suffix.lstrip(".")

        raw_content = None
        if manifest.get_file_modified_at(file_id) == file_stats.st_mtime:
            raw_content = manifest.get_raw_content(file_id)
        if raw_content is None:
            with open(path, "rb") as f:
                raw_content = f.read()

        return SpruceFile(
            file_id=file_id,
            display_name=pathlib.Path(path).name,
            file_type=file_type,
            data_source_id=task.data_source_id,
            raw_content=raw_content,
            chunks=[],
            modified_at=file_stats.st_mtime,
        )
