import asyncio
import pathlib
from collections.abc import Callable
from datetime import datetime, timezone

from ..base import SourceConnector, SUPPORTED_EXTENSIONS

# Google Docs (Google Workspace's native document format) are exported as plain
# text; other Workspace types are not supported. This is distinct from Microsoft
# Word .doc/.docx files, which are downloaded as-is (see _EXTENSION_TO_MIME).
_WORKSPACE_EXPORT_MIME: dict[str, str] = {
    "application/vnd.google-apps.document": "text/plain",
}

# Maps file extensions from SUPPORTED_EXTENSIONS to their Drive MIME types. The
# doc/docx entries are Microsoft Word formats — not Google Docs (see above).
_EXTENSION_TO_MIME: dict[str, str] = {
    "txt":  "text/plain",
    "md":   "text/markdown",
    "html": "text/html",
    "json": "application/json",
    "pdf":  "application/pdf",
    "doc":  "application/msword",  # Microsoft Word 97–2003
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # Microsoft Word
}

_SUPPORTED_MIME_TYPES: frozenset[str] = frozenset(
    mime for ext, mime in _EXTENSION_TO_MIME.items() if ext in SUPPORTED_EXTENSIONS
)


def _build_drive_service(on_token_expired):
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials

    try:
        token = on_token_expired()
    except Exception as exc:
        raise RuntimeError(
            "GoogleDriveSource: on_token_expired() raised an error — "
            "ensure it returns a valid access token string."
        ) from exc
    if not token:
        raise RuntimeError(
            "GoogleDriveSource: on_token_expired() returned an empty token."
        )
    return build("drive", "v3", credentials=Credentials(token=token))


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
        watched_dir: str,
        on_token_expired: Callable[[], str],
        recursive: bool = True,
    ):
        self._folder_id = watched_dir
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

    def is_supported(self, file_identifier: str) -> bool:
        return file_identifier in _WORKSPACE_EXPORT_MIME or file_identifier in _SUPPORTED_MIME_TYPES

    def create_watcher(self, data_source_id: int):
        from spruceup.monitoring.google_drive_watcher import GoogleDriveWatcher

        return GoogleDriveWatcher(
            self._folder_id,
            data_source_id,
            self._on_token_expired,
            self.is_supported,
        )

    async def fetch(self, task, manifest):
        from spruceup.models import SpruceFile

        file_id = task.current_file_id
        service = await asyncio.to_thread(_build_drive_service, self._on_token_expired)

        meta = await asyncio.to_thread(
            service.files().get(
                fileId=file_id,
                fields="name,mimeType,modifiedTime",
            ).execute
        )

        mime_type: str = meta["mimeType"]
        export_mime = _WORKSPACE_EXPORT_MIME.get(mime_type)
        file_type = "txt" if export_mime else pathlib.PurePosixPath(meta["name"]).suffix.lstrip(".")
        modified_at = datetime.fromisoformat(
            meta["modifiedTime"].replace("Z", "+00:00")
        ).timestamp()

        raw_content = None
        if task.use_manifest_cache:
            if manifest.get_file_modified_at(file_id) == modified_at:
                raw_content = manifest.get_raw_content(file_id)

        if raw_content is None:
            if export_mime:
                raw_content = await asyncio.to_thread(
                    service.files().export(
                        fileId=file_id, mimeType=export_mime
                    ).execute
                )
            elif mime_type in _SUPPORTED_MIME_TYPES:
                raw_content = await asyncio.to_thread(
                    service.files().get_media(fileId=file_id).execute
                )
            else:
                raise ValueError(
                    f"Unsupported file type {mime_type!r} for {meta['name']!r} — "
                    f"only Google Docs, plain text, markdown, HTML, JSON, PDF, and "
                    f"Microsoft Word (.doc/.docx) are supported."
                )

        return SpruceFile(
            file_id=file_id,
            display_name=meta["name"],
            file_type=file_type,
            data_source_id=task.data_source_id,
            raw_content=raw_content,
            chunks=[],
            modified_at=modified_at,
        )
