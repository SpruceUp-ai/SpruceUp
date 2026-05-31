"""
Tests for GoogleDriveWatcher._full_scan, ._incremental_scan, ._catch_up, and ._watch.

Google Drive API calls are mocked with MagicMock. asyncio.to_thread is patched
(via the autouse sync_to_thread fixture) to invoke callables synchronously so
tests never spin up real threads or hit network. Real SQLite manifest backed by
a temp file — same pattern as test_local_file_watcher.py.
"""
import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spruceup.manifest import Manifest
from spruceup.models import SyncTask
from spruceup.monitoring.google_drive_watcher import (
    GoogleDriveWatcher,
    _FOLDER_MIME,
    _STATE_FOLDER_IDS,
    _STATE_PAGE_TOKEN,
)
from spruceup.utils.hashing import hash_source_ref


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

WATCHED_FOLDER = "folder-root-id"

SUPPORTED_MIMES = frozenset({
    "text/plain",
    "text/markdown",
    "application/pdf",
    "application/vnd.google-apps.document",
})


def is_supported(mime: str) -> bool:
    return mime in SUPPORTED_MIMES


def drain(queue: asyncio.Queue) -> list[SyncTask]:
    tasks = []
    while not queue.empty():
        tasks.append(queue.get_nowait())
    return tasks


def make_service(
    *,
    start_page_token: str = "token-start",
    list_responses: list[dict] | None = None,
    change_responses: list[dict] | None = None,
) -> MagicMock:
    """
    Build a MagicMock Drive v3 service.

    list_responses: sequential return values for files().list().execute()
    change_responses: sequential return values for changes().list().execute()
    """
    service = MagicMock()
    service.changes.return_value.getStartPageToken.return_value.execute.return_value = {
        "startPageToken": start_page_token
    }
    if list_responses is not None:
        service.files.return_value.list.return_value.execute.side_effect = list_responses
    if change_responses is not None:
        service.changes.return_value.list.return_value.execute.side_effect = change_responses
    return service


def seed_known_ref(manifest: Manifest, ds_id: int, drive_file_id: str) -> None:
    """Insert a minimal file row so the Drive file ID appears in get_source_refs."""
    fid = hash_source_ref(drive_file_id)
    with manifest.connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO files "
            "(id, source_ref, content_hash, data_source_id, file_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (fid, drive_file_id, b"\x00" * 16, ds_id, "txt"),
        )


def set_stored_folder_ids(manifest: Manifest, ds_id: int, folder_ids: list[str]) -> None:
    manifest.set_source_state(ds_id, _STATE_FOLDER_IDS, json.dumps(sorted(folder_ids)))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def manifest(tmp_path):
    return Manifest(str(tmp_path / "manifest.db"))


@pytest.fixture
def ds_id(manifest):
    return manifest.register_source("google_drive", WATCHED_FOLDER)


@pytest.fixture
def watcher(ds_id):
    return GoogleDriveWatcher(
        WATCHED_FOLDER,
        ds_id,
        "google_drive",
        on_token_expired=lambda: "fake-token",
        is_supported=is_supported,
    )


@pytest.fixture(autouse=True)
def sync_to_thread():
    """Patch asyncio.to_thread to invoke callables synchronously for all tests."""
    async def _fake(func, *args, **kwargs):
        return func(*args, **kwargs)
    with patch("asyncio.to_thread", _fake):
        yield


# ---------------------------------------------------------------------------
# _full_scan
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_scan_empty_folder_saves_token(watcher, manifest):
    service = make_service(
        start_page_token="tok-42",
        list_responses=[{"files": []}],
    )
    queue = asyncio.Queue()
    await watcher._full_scan(service, queue, manifest)

    assert drain(queue) == []
    assert manifest.get_source_state(watcher._data_source_id, _STATE_PAGE_TOKEN) == "tok-42"


@pytest.mark.asyncio
async def test_full_scan_supported_file_emits_upsert(watcher, manifest):
    service = make_service(
        list_responses=[{"files": [{"id": "file-abc", "mimeType": "text/plain"}]}],
    )
    queue = asyncio.Queue()
    await watcher._full_scan(service, queue, manifest)

    tasks = drain(queue)
    assert len(tasks) == 1
    assert tasks[0].change_type == "upsert"
    assert tasks[0].identifier == "file-abc"


@pytest.mark.asyncio
async def test_full_scan_unsupported_mime_not_emitted(watcher, manifest):
    service = make_service(
        list_responses=[{"files": [{"id": "file-xyz", "mimeType": "image/png"}]}],
    )
    queue = asyncio.Queue()
    await watcher._full_scan(service, queue, manifest)
    assert drain(queue) == []


@pytest.mark.asyncio
async def test_full_scan_subfolder_is_recursed(watcher, manifest):
    # Root page returns a subfolder; subfolder page returns a supported file.
    service = make_service(
        list_responses=[
            {"files": [{"id": "sub-folder-id", "mimeType": _FOLDER_MIME}]},
            {"files": [{"id": "nested-file", "mimeType": "text/plain"}]},
        ],
    )
    queue = asyncio.Queue()
    await watcher._full_scan(service, queue, manifest)

    tasks = drain(queue)
    assert len(tasks) == 1
    assert tasks[0].identifier == "nested-file"

    stored = json.loads(manifest.get_source_state(watcher._data_source_id, _STATE_FOLDER_IDS))
    assert "sub-folder-id" in stored


@pytest.mark.asyncio
async def test_full_scan_multi_page_drains_all_files(watcher, manifest):
    service = make_service(
        list_responses=[
            {"files": [{"id": "file-1", "mimeType": "text/plain"}], "nextPageToken": "p2"},
            {"files": [{"id": "file-2", "mimeType": "text/plain"}]},
        ],
    )
    queue = asyncio.Queue()
    await watcher._full_scan(service, queue, manifest)

    ids = {t.identifier for t in drain(queue)}
    assert ids == {"file-1", "file-2"}


@pytest.mark.asyncio
async def test_full_scan_token_anchored_before_bfs(watcher, manifest):
    """getStartPageToken must be called before any files().list() to avoid missing
    changes that race with the scan."""
    call_log: list[str] = []

    service = MagicMock()
    service.changes.return_value.getStartPageToken.return_value.execute.side_effect = (
        lambda: call_log.append("get_token") or {"startPageToken": "tok"}
    )
    service.files.return_value.list.return_value.execute.side_effect = (
        lambda: call_log.append("list_files") or {"files": []}
    )

    await watcher._full_scan(service, asyncio.Queue(), manifest)

    assert call_log[0] == "get_token", "getStartPageToken must precede files().list()"


# ---------------------------------------------------------------------------
# _incremental_scan
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_incremental_scan_removed_known_file_emits_delete(watcher, manifest, ds_id):
    seed_known_ref(manifest, ds_id, "file-del")
    service = make_service(change_responses=[{
        "changes": [{"fileId": "file-del", "removed": True, "file": {}}],
        "newStartPageToken": "tok-next",
    }])
    queue = asyncio.Queue()
    await watcher._incremental_scan(service, queue, manifest, stored_token="tok-old")

    tasks = drain(queue)
    assert len(tasks) == 1
    assert tasks[0].change_type == "delete"
    assert tasks[0].identifier == "file-del"


@pytest.mark.asyncio
async def test_incremental_scan_trashed_known_file_emits_delete(watcher, manifest, ds_id):
    seed_known_ref(manifest, ds_id, "file-trash")
    service = make_service(change_responses=[{
        "changes": [{
            "fileId": "file-trash",
            "removed": False,
            "file": {
                "id": "file-trash",
                "parents": [WATCHED_FOLDER],
                "trashed": True,
                "mimeType": "text/plain",
            },
        }],
        "newStartPageToken": "tok-next",
    }])
    queue = asyncio.Queue()
    await watcher._incremental_scan(service, queue, manifest, stored_token="tok-old")

    tasks = drain(queue)
    assert len(tasks) == 1
    assert tasks[0].change_type == "delete"


@pytest.mark.asyncio
async def test_incremental_scan_new_file_in_tree_emits_upsert(watcher, manifest):
    service = make_service(change_responses=[{
        "changes": [{
            "fileId": "file-new",
            "removed": False,
            "file": {
                "id": "file-new",
                "parents": [WATCHED_FOLDER],
                "trashed": False,
                "mimeType": "text/plain",
            },
        }],
        "newStartPageToken": "tok-next",
    }])
    queue = asyncio.Queue()
    await watcher._incremental_scan(service, queue, manifest, stored_token="tok-old")

    tasks = drain(queue)
    assert len(tasks) == 1
    assert tasks[0].change_type == "upsert"
    assert tasks[0].identifier == "file-new"


@pytest.mark.asyncio
async def test_incremental_scan_known_file_updated_in_tree_emits_upsert(watcher, manifest, ds_id):
    seed_known_ref(manifest, ds_id, "file-upd")
    service = make_service(change_responses=[{
        "changes": [{
            "fileId": "file-upd",
            "removed": False,
            "file": {
                "id": "file-upd",
                "parents": [WATCHED_FOLDER],
                "trashed": False,
                "mimeType": "text/plain",
            },
        }],
        "newStartPageToken": "tok-next",
    }])
    queue = asyncio.Queue()
    await watcher._incremental_scan(service, queue, manifest, stored_token="tok-old")

    tasks = drain(queue)
    assert len(tasks) == 1
    assert tasks[0].change_type == "upsert"


@pytest.mark.asyncio
async def test_incremental_scan_known_file_moved_out_of_tree_emits_delete(watcher, manifest, ds_id):
    seed_known_ref(manifest, ds_id, "file-moved")
    service = make_service(change_responses=[{
        "changes": [{
            "fileId": "file-moved",
            "removed": False,
            "file": {
                "id": "file-moved",
                "parents": ["other-folder"],  # outside the watched tree
                "trashed": False,
                "mimeType": "text/plain",
            },
        }],
        "newStartPageToken": "tok-next",
    }])
    queue = asyncio.Queue()
    await watcher._incremental_scan(service, queue, manifest, stored_token="tok-old")

    tasks = drain(queue)
    assert len(tasks) == 1
    assert tasks[0].change_type == "delete"


@pytest.mark.asyncio
async def test_incremental_scan_known_file_mime_changed_to_unsupported_emits_delete(
    watcher, manifest, ds_id
):
    seed_known_ref(manifest, ds_id, "file-mime")
    service = make_service(change_responses=[{
        "changes": [{
            "fileId": "file-mime",
            "removed": False,
            "file": {
                "id": "file-mime",
                "parents": [WATCHED_FOLDER],
                "trashed": False,
                "mimeType": "image/png",  # was text/plain, now unsupported
            },
        }],
        "newStartPageToken": "tok-next",
    }])
    queue = asyncio.Queue()
    await watcher._incremental_scan(service, queue, manifest, stored_token="tok-old")

    tasks = drain(queue)
    assert len(tasks) == 1
    assert tasks[0].change_type == "delete"


@pytest.mark.asyncio
async def test_incremental_scan_new_file_not_in_tree_not_emitted(watcher, manifest):
    service = make_service(change_responses=[{
        "changes": [{
            "fileId": "file-outside",
            "removed": False,
            "file": {
                "id": "file-outside",
                "parents": ["other-folder"],
                "trashed": False,
                "mimeType": "text/plain",
            },
        }],
        "newStartPageToken": "tok-next",
    }])
    queue = asyncio.Queue()
    await watcher._incremental_scan(service, queue, manifest, stored_token="tok-old")
    assert drain(queue) == []


@pytest.mark.asyncio
async def test_incremental_scan_new_subfolder_in_tree_saved_to_manifest(watcher, manifest):
    service = make_service(change_responses=[{
        "changes": [{
            "fileId": "sub-new",
            "removed": False,
            "file": {
                "id": "sub-new",
                "parents": [WATCHED_FOLDER],
                "trashed": False,
                "mimeType": _FOLDER_MIME,
            },
        }],
        "newStartPageToken": "tok-next",
    }])
    queue = asyncio.Queue()
    await watcher._incremental_scan(service, queue, manifest, stored_token="tok-old")

    assert drain(queue) == []
    stored = json.loads(manifest.get_source_state(watcher._data_source_id, _STATE_FOLDER_IDS))
    assert "sub-new" in stored


@pytest.mark.asyncio
async def test_incremental_scan_removed_subfolder_pruned_from_manifest(watcher, manifest, ds_id):
    set_stored_folder_ids(manifest, ds_id, [WATCHED_FOLDER, "sub-old"])
    service = make_service(change_responses=[{
        "changes": [{"fileId": "sub-old", "removed": True, "file": {}}],
        "newStartPageToken": "tok-next",
    }])
    queue = asyncio.Queue()
    await watcher._incremental_scan(service, queue, manifest, stored_token="tok-old")

    stored = json.loads(manifest.get_source_state(watcher._data_source_id, _STATE_FOLDER_IDS))
    assert "sub-old" not in stored


@pytest.mark.asyncio
async def test_incremental_scan_multi_page_drains_all_changes(watcher, manifest):
    service = make_service(change_responses=[
        {
            "changes": [{
                "fileId": "file-p1", "removed": False,
                "file": {"id": "file-p1", "parents": [WATCHED_FOLDER], "trashed": False, "mimeType": "text/plain"},
            }],
            "nextPageToken": "page2",
        },
        {
            "changes": [{
                "fileId": "file-p2", "removed": False,
                "file": {"id": "file-p2", "parents": [WATCHED_FOLDER], "trashed": False, "mimeType": "text/plain"},
            }],
            "newStartPageToken": "tok-final",
        },
    ])
    queue = asyncio.Queue()
    await watcher._incremental_scan(service, queue, manifest, stored_token="tok-old")

    ids = {t.identifier for t in drain(queue)}
    assert ids == {"file-p1", "file-p2"}


@pytest.mark.asyncio
async def test_incremental_scan_saves_new_start_page_token(watcher, manifest):
    service = make_service(change_responses=[{
        "changes": [],
        "newStartPageToken": "tok-new",
    }])
    await watcher._incremental_scan(service, asyncio.Queue(), manifest, stored_token="tok-old")

    assert manifest.get_source_state(watcher._data_source_id, _STATE_PAGE_TOKEN) == "tok-new"


# ---------------------------------------------------------------------------
# _catch_up
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_catch_up_no_stored_token_calls_full_scan(watcher, manifest):
    with patch.object(watcher, "_full_scan", new_callable=AsyncMock, return_value=0) as mock_full, \
         patch.object(watcher, "_incremental_scan", new_callable=AsyncMock) as mock_inc, \
         patch.object(watcher, "_build_service", return_value=MagicMock()):
        await watcher._catch_up(asyncio.Queue(), manifest)

    mock_full.assert_called_once()
    mock_inc.assert_not_called()


@pytest.mark.asyncio
async def test_catch_up_stored_token_calls_incremental_scan(watcher, manifest, ds_id):
    manifest.set_source_state(ds_id, _STATE_PAGE_TOKEN, "existing-token")

    with patch.object(watcher, "_full_scan", new_callable=AsyncMock) as mock_full, \
         patch.object(watcher, "_incremental_scan", new_callable=AsyncMock, return_value=(0, 0)) as mock_inc, \
         patch.object(watcher, "_build_service", return_value=MagicMock()):
        await watcher._catch_up(asyncio.Queue(), manifest)

    mock_inc.assert_called_once()
    mock_full.assert_not_called()


@pytest.mark.asyncio
async def test_catch_up_force_reindex_calls_full_scan_even_with_stored_token(watcher, manifest, ds_id):
    manifest.set_source_state(ds_id, _STATE_PAGE_TOKEN, "existing-token")

    with patch.object(watcher, "_full_scan", new_callable=AsyncMock, return_value=0) as mock_full, \
         patch.object(watcher, "_incremental_scan", new_callable=AsyncMock) as mock_inc, \
         patch.object(watcher, "_build_service", return_value=MagicMock()):
        await watcher._catch_up(asyncio.Queue(), manifest, force_reindex=True)

    mock_full.assert_called_once()
    mock_inc.assert_not_called()


# ---------------------------------------------------------------------------
# _watch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_watch_polls_incremental_scan_after_catchup_done(watcher, manifest, ds_id):
    manifest.set_source_state(ds_id, _STATE_PAGE_TOKEN, "tok")
    catchup_done = asyncio.Event()
    catchup_done.set()

    mock_scan = AsyncMock(side_effect=[(1, 0), asyncio.CancelledError()])

    with patch.object(watcher, "_incremental_scan", mock_scan), \
         patch.object(watcher, "_build_service", return_value=MagicMock()), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(asyncio.CancelledError):
            await watcher._watch(asyncio.Queue(), manifest, catchup_done)

    assert mock_scan.call_count == 2


@pytest.mark.asyncio
async def test_watch_401_refreshes_service_and_retries(watcher, manifest, ds_id):
    from googleapiclient.errors import HttpError

    manifest.set_source_state(ds_id, _STATE_PAGE_TOKEN, "tok")
    catchup_done = asyncio.Event()
    catchup_done.set()

    mock_resp = MagicMock()
    mock_resp.status = 401
    auth_err = HttpError(mock_resp, b"Unauthorized")

    mock_scan = AsyncMock(side_effect=[auth_err, asyncio.CancelledError()])
    mock_build = MagicMock(return_value=MagicMock())

    with patch.object(watcher, "_incremental_scan", mock_scan), \
         patch.object(watcher, "_build_service", mock_build), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(asyncio.CancelledError):
            await watcher._watch(asyncio.Queue(), manifest, catchup_done)

    assert mock_build.call_count == 2   # initial build + refresh after 401
    assert mock_scan.call_count == 2    # first attempt (401) + retry after refresh
