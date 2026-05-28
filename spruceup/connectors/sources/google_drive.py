from collections.abc import Callable

from ..base import SourceConnector


class GoogleDriveSource(SourceConnector):
    def __init__(
        self,
        folder_id: str,
        on_token_expired: Callable[[], str],
        recursive: bool = True,
    ):
        self._folder_id = folder_id
        self._on_token_expired = on_token_expired
        self._recursive = recursive

    @property
    def source_type(self) -> str:
        return "google_drive"

    @property
    def source_identifier(self) -> str:
        return self.folder_id

    def create_watcher(self, data_source_id: int):
        from spruceup.monitoring.google_drive_watcher import GoogleDriveWatcher

        return GoogleDriveWatcher(
            self._folder_id,
            data_source_id,
            self.source_type,
            self._on_token_expired,
        )

    def display_name(self, identifier: str) -> str: ...

    def decode_content(self, raw_content: bytes) -> str: ...

    async def fetch(self, task): ...
