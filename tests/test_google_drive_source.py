"""
Tests for GoogleDriveSource, _build_drive_service, and _folder_is_ancestor.

asyncio.to_thread is patched (autouse sync_to_thread fixture) to invoke
callables synchronously — same pattern as test_google_drive_watcher.py.
All Google API I/O is replaced with MagicMock; no network calls are made.
"""

import asyncio
import hashlib
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spruceup.connectors.sources.google_drive import (
    GoogleDriveSource,
    _build_drive_service,
    _folder_is_ancestor,
    _WORKSPACE_EXPORT_MIME,
    _SUPPORTED_MIME_TYPES,
)
from spruceup.models import SpruceFile, SyncTask
from spruceup.utils.hashing import hash_source_ref


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

FOLDER_ID = "folder-root-id"


def blake2b(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=16).digest()


def make_task(identifier: str, data_source_id: int = 1) -> SyncTask:
    return SyncTask(
        source_type="google_drive",
        identifier=identifier,
        change_type="upsert",
        data_source_id=data_source_id,
    )


def make_source(folder_id: str = FOLDER_ID) -> GoogleDriveSource:
    return GoogleDriveSource(
        watched_dir=folder_id,
        on_token_expired=lambda: "fake-token",
    )


def ancestor_service(parents_sequence: list[list[str]]) -> MagicMock:
    """Build a service mock whose files().get().execute() returns parent lists
    in sequence across successive calls, simulating a folder hierarchy walk."""
    service = MagicMock()
    service.files.return_value.get.return_value.execute.side_effect = [
        {"parents": p} for p in parents_sequence
    ]
    return service


def fetch_service(
    *,
    mime_type: str,
    file_name: str = "document",
    modified_time: str = "2024-01-15T10:30:00Z",
    raw_content: bytes = b"file content",
) -> MagicMock:
    """Build a service mock for fetch() tests."""
    service = MagicMock()
    service.files.return_value.get.return_value.execute.return_value = {
        "name": file_name,
        "mimeType": mime_type,
        "modifiedTime": modified_time,
    }
    service.files.return_value.export.return_value.execute.return_value = raw_content
    service.files.return_value.get_media.return_value.execute.return_value = raw_content
    return service


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def source():
    return make_source()


@pytest.fixture(autouse=True)
def sync_to_thread():
    """Patch asyncio.to_thread to invoke callables synchronously for all tests."""
    async def _fake(func, *args, **kwargs):
        return func(*args, **kwargs)
    with patch("asyncio.to_thread", _fake):
        yield


# ---------------------------------------------------------------------------
# _build_drive_service
# ---------------------------------------------------------------------------

def test_build_drive_service_returns_service_for_valid_token():
    mock_svc = MagicMock()
    with patch("googleapiclient.discovery.build", return_value=mock_svc), \
         patch("google.oauth2.credentials.Credentials"):
        result = _build_drive_service(lambda: "valid-token")
    assert result is mock_svc


def test_build_drive_service_raises_if_callback_raises():
    def bad_callback():
        raise Exception("auth failed")
    with pytest.raises(RuntimeError, match="raised an error"):
        _build_drive_service(bad_callback)


def test_build_drive_service_raises_if_token_is_empty_string():
    with pytest.raises(RuntimeError, match="empty token"):
        _build_drive_service(lambda: "")


def test_build_drive_service_raises_if_token_is_none():
    with pytest.raises(RuntimeError, match="empty token"):
        _build_drive_service(lambda: None)


# ---------------------------------------------------------------------------
# _folder_is_ancestor
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_folder_is_ancestor_direct_parent_returns_true():
    service = ancestor_service([["ancestor-id", "other-parent"]])
    assert await _folder_is_ancestor(service, "ancestor-id", "folder-id", set()) is True


@pytest.mark.asyncio
async def test_folder_is_ancestor_two_levels_up_returns_true():
    # folder-id → parent-id → ancestor-id
    service = ancestor_service([["parent-id"], ["ancestor-id"]])
    assert await _folder_is_ancestor(service, "ancestor-id", "folder-id", set()) is True


@pytest.mark.asyncio
async def test_folder_is_ancestor_unrelated_returns_false():
    # folder-id → unrelated-parent → no more parents
    service = ancestor_service([["unrelated-parent"], []])
    assert await _folder_is_ancestor(service, "ancestor-id", "folder-id", set()) is False


@pytest.mark.asyncio
async def test_folder_is_ancestor_empty_parents_returns_false():
    service = ancestor_service([[]])
    assert await _folder_is_ancestor(service, "ancestor-id", "folder-id", set()) is False


@pytest.mark.asyncio
async def test_folder_is_ancestor_api_exception_returns_false():
    service = MagicMock()
    service.files.return_value.get.return_value.execute.side_effect = Exception("API error")
    assert await _folder_is_ancestor(service, "ancestor-id", "folder-id", set()) is False


@pytest.mark.asyncio
async def test_folder_is_ancestor_stops_at_known_root():
    # First parent is a known root — stops early and returns False.
    service = ancestor_service([["known-root"]])
    result = await _folder_is_ancestor(
        service, "ancestor-id", "folder-id", known_roots={"known-root"}
    )
    assert result is False


# ---------------------------------------------------------------------------
# GoogleDriveSource — properties
# ---------------------------------------------------------------------------

def test_source_type_is_google_drive(source):
    assert source.source_type == "google_drive"


def test_source_identifier_returns_folder_id(source):
    assert source.source_identifier == FOLDER_ID


# ---------------------------------------------------------------------------
# GoogleDriveSource — validate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_single_source_passes():
    await GoogleDriveSource.validate([make_source()])


@pytest.mark.asyncio
async def test_validate_two_non_nested_folders_passes():
    with patch(
        "spruceup.connectors.sources.google_drive._build_drive_service",
        return_value=MagicMock(),
    ), patch(
        "spruceup.connectors.sources.google_drive._folder_is_ancestor",
        new_callable=AsyncMock,
        return_value=False,
    ):
        await GoogleDriveSource.validate([make_source("folder-a"), make_source("folder-b")])


@pytest.mark.asyncio
async def test_validate_same_folder_id_raises():
    with pytest.raises(ValueError, match="same folder"):
        await GoogleDriveSource.validate([make_source("folder-x"), make_source("folder-x")])


@pytest.mark.asyncio
async def test_validate_same_folder_error_includes_folder_id():
    with pytest.raises(ValueError) as exc_info:
        await GoogleDriveSource.validate([make_source("folder-x"), make_source("folder-x")])
    assert "folder-x" in str(exc_info.value)


@pytest.mark.asyncio
async def test_validate_src_a_ancestor_of_src_b_raises():
    with patch(
        "spruceup.connectors.sources.google_drive._build_drive_service",
        return_value=MagicMock(),
    ), patch(
        "spruceup.connectors.sources.google_drive._folder_is_ancestor",
        new_callable=AsyncMock,
        return_value=True,
    ):
        with pytest.raises(ValueError, match="ancestor"):
            await GoogleDriveSource.validate([make_source("folder-a"), make_source("folder-b")])


@pytest.mark.asyncio
async def test_validate_src_b_ancestor_of_src_a_raises():
    # First _folder_is_ancestor call (src_a is ancestor of src_b) returns False;
    # second call (src_b is ancestor of src_a) returns True.
    call_count = {"n": 0}

    async def fake_is_ancestor(service, ancestor_id, folder_id, known_roots):
        call_count["n"] += 1
        return call_count["n"] == 2

    with patch(
        "spruceup.connectors.sources.google_drive._build_drive_service",
        return_value=MagicMock(),
    ), patch(
        "spruceup.connectors.sources.google_drive._folder_is_ancestor",
        fake_is_ancestor,
    ):
        with pytest.raises(ValueError, match="ancestor"):
            await GoogleDriveSource.validate([make_source("folder-a"), make_source("folder-b")])


# ---------------------------------------------------------------------------
# GoogleDriveSource — is_supported
# ---------------------------------------------------------------------------

def test_is_supported_workspace_mime_types_return_true(source):
    for mime in _WORKSPACE_EXPORT_MIME:
        assert source.is_supported(mime) is True


def test_is_supported_regular_supported_mimes_return_true(source):
    for mime in _SUPPORTED_MIME_TYPES:
        assert source.is_supported(mime) is True


@pytest.mark.parametrize("mime", ["image/png", "video/mp4", "application/octet-stream"])
def test_is_supported_unsupported_mime_returns_false(source, mime):
    assert source.is_supported(mime) is False


# ---------------------------------------------------------------------------
# GoogleDriveSource — create_watcher
# ---------------------------------------------------------------------------

def test_create_watcher_returns_google_drive_watcher(source):
    from spruceup.monitoring.google_drive_watcher import GoogleDriveWatcher
    assert isinstance(source.create_watcher(data_source_id=5), GoogleDriveWatcher)


# ---------------------------------------------------------------------------
# GoogleDriveSource — display_name
# ---------------------------------------------------------------------------

def test_display_name_returns_identifier_unchanged(source):
    assert source.display_name("file-abc-123") == "file-abc-123"


# ---------------------------------------------------------------------------
# GoogleDriveSource — decode_content
# ---------------------------------------------------------------------------

def test_decode_content_decodes_utf8(source):
    assert source.decode_content(b"hello world") == "hello world"


def test_decode_content_replaces_invalid_bytes(source):
    result = source.decode_content(b"valid \xff\xfe bytes")
    assert isinstance(result, str)
    assert "valid" in result


# ---------------------------------------------------------------------------
# GoogleDriveSource — fetch (Google Doc export path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_google_doc_returns_spruce_file(source):
    svc = fetch_service(mime_type="application/vnd.google-apps.document")
    with patch("spruceup.connectors.sources.google_drive._build_drive_service", return_value=svc):
        result = await source.fetch(make_task("file-doc-id"))
    assert isinstance(result, SpruceFile)


@pytest.mark.asyncio
async def test_fetch_google_doc_file_type_is_txt(source):
    svc = fetch_service(mime_type="application/vnd.google-apps.document")
    with patch("spruceup.connectors.sources.google_drive._build_drive_service", return_value=svc):
        result = await source.fetch(make_task("file-doc-id"))
    assert result.file_type == "txt"


@pytest.mark.asyncio
async def test_fetch_google_doc_uses_export_not_get_media(source):
    raw = b"exported doc"
    svc = fetch_service(mime_type="application/vnd.google-apps.document", raw_content=raw)
    with patch("spruceup.connectors.sources.google_drive._build_drive_service", return_value=svc):
        result = await source.fetch(make_task("file-doc-id"))
    assert result.raw_content == raw
    svc.files.return_value.export.assert_called_once()
    svc.files.return_value.get_media.assert_not_called()


# ---------------------------------------------------------------------------
# GoogleDriveSource — fetch (regular file path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_text_file_returns_spruce_file(source):
    svc = fetch_service(mime_type="text/plain", file_name="notes.txt")
    with patch("spruceup.connectors.sources.google_drive._build_drive_service", return_value=svc):
        result = await source.fetch(make_task("file-txt-id"))
    assert isinstance(result, SpruceFile)


@pytest.mark.asyncio
async def test_fetch_text_file_uses_get_media_not_export(source):
    raw = b"plain text content"
    svc = fetch_service(mime_type="text/plain", file_name="notes.txt", raw_content=raw)
    with patch("spruceup.connectors.sources.google_drive._build_drive_service", return_value=svc):
        result = await source.fetch(make_task("file-txt-id"))
    assert result.raw_content == raw
    svc.files.return_value.get_media.assert_called_once()
    svc.files.return_value.export.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_file_type_derived_from_filename_txt(source):
    svc = fetch_service(mime_type="text/plain", file_name="notes.txt")
    with patch("spruceup.connectors.sources.google_drive._build_drive_service", return_value=svc):
        result = await source.fetch(make_task("file-txt-id"))
    assert result.file_type == "txt"


@pytest.mark.asyncio
async def test_fetch_file_type_derived_from_filename_md(source):
    svc = fetch_service(mime_type="text/markdown", file_name="readme.md")
    with patch("spruceup.connectors.sources.google_drive._build_drive_service", return_value=svc):
        result = await source.fetch(make_task("file-md-id"))
    assert result.file_type == "md"


# ---------------------------------------------------------------------------
# GoogleDriveSource — fetch (SpruceFile field assertions)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_file_id_matches_hash_source_ref(source):
    svc = fetch_service(mime_type="text/plain", file_name="doc.txt")
    with patch("spruceup.connectors.sources.google_drive._build_drive_service", return_value=svc):
        result = await source.fetch(make_task("file-abc"))
    assert result.file_id == hash_source_ref("file-abc")


@pytest.mark.asyncio
async def test_fetch_source_ref_is_drive_file_id(source):
    svc = fetch_service(mime_type="text/plain", file_name="doc.txt")
    with patch("spruceup.connectors.sources.google_drive._build_drive_service", return_value=svc):
        result = await source.fetch(make_task("file-abc"))
    assert result.source_ref == "file-abc"


@pytest.mark.asyncio
async def test_fetch_display_name_is_drive_file_name(source):
    svc = fetch_service(mime_type="text/plain", file_name="my-document.txt")
    with patch("spruceup.connectors.sources.google_drive._build_drive_service", return_value=svc):
        result = await source.fetch(make_task("file-abc"))
    assert result.display_name == "my-document.txt"


@pytest.mark.asyncio
async def test_fetch_content_hash_is_blake2b_of_raw_content(source):
    raw = b"the file bytes"
    svc = fetch_service(mime_type="text/plain", file_name="doc.txt", raw_content=raw)
    with patch("spruceup.connectors.sources.google_drive._build_drive_service", return_value=svc):
        result = await source.fetch(make_task("file-abc"))
    assert result.content_hash == blake2b(raw)


@pytest.mark.asyncio
async def test_fetch_data_source_id_propagated_from_task(source):
    svc = fetch_service(mime_type="text/plain", file_name="doc.txt")
    with patch("spruceup.connectors.sources.google_drive._build_drive_service", return_value=svc):
        result = await source.fetch(make_task("file-abc", data_source_id=77))
    assert result.data_source_id == 77


@pytest.mark.asyncio
async def test_fetch_chunks_starts_empty(source):
    svc = fetch_service(mime_type="text/plain", file_name="doc.txt")
    with patch("spruceup.connectors.sources.google_drive._build_drive_service", return_value=svc):
        result = await source.fetch(make_task("file-abc"))
    assert result.chunks == []


@pytest.mark.asyncio
async def test_fetch_modified_at_parsed_from_modified_time(source):
    svc = fetch_service(mime_type="text/plain", file_name="doc.txt", modified_time="2024-06-15T08:00:00Z")
    with patch("spruceup.connectors.sources.google_drive._build_drive_service", return_value=svc):
        result = await source.fetch(make_task("file-abc"))
    expected = datetime(2024, 6, 15, 8, 0, 0, tzinfo=timezone.utc).timestamp()
    assert result.source_metadata["modified_at"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# GoogleDriveSource — fetch (unsupported MIME type)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_unsupported_mime_raises_value_error(source):
    svc = fetch_service(mime_type="image/png", file_name="photo.png")
    with patch("spruceup.connectors.sources.google_drive._build_drive_service", return_value=svc):
        with pytest.raises(ValueError, match="Unsupported file type"):
            await source.fetch(make_task("file-img-id"))
