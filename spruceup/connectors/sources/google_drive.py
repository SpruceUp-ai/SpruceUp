import asyncio
from collections.abc import Callable

from ..base import SourceConnector


def _build_drive_service(on_token_expired):
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials

    return build("drive", "v3", credentials=Credentials(token=on_token_expired()))


async def _folder_is_ancestor(
    service, ancestor_id: str, folder_id: str, known_roots: set[str]
) -> bool:
    """Walk folder_id's parent chain; return True if ancestor_id appears in it.

    Stops early if current climbs past any known watched root, since no
    ancestor relationship is possible beyond that point.
    """
    current = folder_id
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        try:
            meta = await asyncio.to_thread(
                service.files().get(fileId=current, fields="parents").execute
            )
        except Exception:
            return False
        parents = meta.get("parents") or []
        if ancestor_id in parents:
            return True
        if not parents:
            return False
        current = parents[0]
        if current in known_roots:
            return False
    return False


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
        return self._folder_id

    @classmethod
    async def validate(cls, sources: list["GoogleDriveSource"]) -> None:
        known_roots = {src._folder_id for src in sources}
        for i, src_a in enumerate(sources):
            for src_b in sources[i + 1:]:
                if src_a._folder_id == src_b._folder_id:
                    raise ValueError(
                        f"Two GoogleDriveSource instances watch the same folder "
                        f"({src_a._folder_id!r})."
                    )
                service = await asyncio.to_thread(
                    _build_drive_service, src_a._on_token_expired
                )
                if await _folder_is_ancestor(
                    service, src_a._folder_id, src_b._folder_id, known_roots
                ):
                    raise ValueError(
                        f"GoogleDriveSource {src_a._folder_id!r} is an ancestor of "
                        f"{src_b._folder_id!r}. Nested watched directories cause duplicate processing."
                    )
                if await _folder_is_ancestor(
                    service, src_b._folder_id, src_a._folder_id, known_roots
                ):
                    raise ValueError(
                        f"GoogleDriveSource {src_b._folder_id!r} is an ancestor of "
                        f"{src_a._folder_id!r}. Nested watched directories cause duplicate processing."
                    )

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
