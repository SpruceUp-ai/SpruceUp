from dataclasses import dataclass

from ..base import SourceConnector


@dataclass
class LocalFilesSource(SourceConnector):
    watched_dir: str

    def create_watcher(self):
        from spruceup.monitoring.monitor import LocalFileWatcher
        return LocalFileWatcher(self.watched_dir)
