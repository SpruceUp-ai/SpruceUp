"""
Tests for LocalFileWatcher._catch_up and ._watch.

Both sections use a real SQLite manifest backed by a temp file. File I/O uses
real temp files so stat() calls (inode, mtime) are genuine. awatch is patched
for _watch tests so they don't block waiting for real filesystem events.
"""

import asyncio
import hashlib
import logging
import pathlib
from unittest.mock import patch

import pytest
from watchfiles import Change

from spruceup.manifest import Manifest
from spruceup.models import SyncTask
from spruceup.monitoring.local_file_watcher import LocalFileWatcher
from spruceup.utils.hashing import hash_source_ref


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUPPORTED = frozenset({"txt", "md", "pdf"})


def is_supported(path: str) -> bool:
    return pathlib.Path(path).suffix.lstrip(".").lower() in SUPPORTED


def blake2b(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=16).digest()


def drain(queue: asyncio.Queue) -> list[SyncTask]:
    tasks = []
    while not queue.empty():
        tasks.append(queue.get_nowait())
    return tasks


def seed_file(
    manifest: Manifest,
    path: pathlib.Path,
    ds_id: int,
    *,
    content: bytes = b"hello",
    mtime: float | None = None,
    inode: int | None = None,
) -> bytes:
    """Insert a file record into the manifest. Uses real stat unless overridden."""
    stat = path.stat()
    file_id = hash_source_ref(str(path))
    metadata = {
        "inode": inode if inode is not None else stat.st_ino,
        "mtime": mtime if mtime is not None else stat.st_mtime,
    }
    with manifest.transaction():
        manifest._conn.execute(
            "INSERT OR REPLACE INTO files "
            "(id, source_ref, content_hash, data_source_id, file_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (file_id, str(path), blake2b(content), ds_id, path.suffix.lstrip(".")),
        )
        manifest.upsert_file_metadata(file_id, metadata)
    return file_id


def seed_file_no_mtime(
    manifest: Manifest,
    path: pathlib.Path,
    ds_id: int,
    *,
    content: bytes = b"hello",
) -> bytes:
    """Seed a file record with only inode in metadata — simulates a record from
    before mtime tracking was added."""
    stat = path.stat()
    file_id = hash_source_ref(str(path))
    with manifest.transaction():
        manifest._conn.execute(
            "INSERT OR REPLACE INTO files "
            "(id, source_ref, content_hash, data_source_id, file_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (file_id, str(path), blake2b(content), ds_id, path.suffix.lstrip(".")),
        )
        manifest.upsert_file_metadata(file_id, {"inode": stat.st_ino})
    return file_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    return d


@pytest.fixture
def manifest(tmp_path):
    return Manifest(str(tmp_path / "manifest.db"))


@pytest.fixture
def ds_id(manifest, corpus):
    return manifest.register_source("local", str(corpus))


@pytest.fixture
def watcher(corpus, ds_id):
    return LocalFileWatcher(str(corpus), ds_id, "local", is_supported)


# ---------------------------------------------------------------------------
# _catch_up — task emission
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_catch_up_new_file_emits_upsert(watcher, manifest, corpus):
    (corpus / "doc.txt").write_bytes(b"content")
    queue = asyncio.Queue()
    await watcher._catch_up(queue, manifest)
    tasks = drain(queue)
    assert len(tasks) == 1
    assert tasks[0].change_type == "upsert"
    assert tasks[0].identifier == str(corpus / "doc.txt")


@pytest.mark.asyncio
async def test_catch_up_force_reindex_upserts_known_file(watcher, manifest, corpus, ds_id):
    path = corpus / "doc.txt"
    path.write_bytes(b"content")
    seed_file(manifest, path, ds_id, content=b"content")
    queue = asyncio.Queue()
    await watcher._catch_up(queue, manifest, force_reindex=True)
    tasks = drain(queue)
    assert len(tasks) == 1
    assert tasks[0].change_type == "upsert"


@pytest.mark.asyncio
async def test_catch_up_unchanged_file_emits_no_task(watcher, manifest, corpus, ds_id):
    path = corpus / "doc.txt"
    path.write_bytes(b"content")
    seed_file(manifest, path, ds_id, content=b"content")
    queue = asyncio.Queue()
    await watcher._catch_up(queue, manifest)
    assert drain(queue) == []


@pytest.mark.asyncio
async def test_catch_up_renamed_file_emits_move(watcher, manifest, corpus, ds_id):
    new_path = corpus / "new_name.txt"
    new_path.write_bytes(b"content")
    stat = new_path.stat()
    old_path_str = str(corpus / "old_name.txt")

    # Seed manifest with the old path but use the real file's inode and mtime so the
    # mtime fast-path fires and only the path difference triggers a move.
    file_id = hash_source_ref(old_path_str)
    with manifest.transaction():
        manifest._conn.execute(
            "INSERT OR REPLACE INTO files "
            "(id, source_ref, content_hash, data_source_id, file_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (file_id, old_path_str, blake2b(b"content"), ds_id, "txt"),
        )
        manifest.upsert_file_metadata(file_id, {
            "inode": stat.st_ino,
            "mtime": stat.st_mtime,
        })

    queue = asyncio.Queue()
    await watcher._catch_up(queue, manifest)
    tasks = drain(queue)
    assert len(tasks) == 1
    assert tasks[0].change_type == "move"
    assert tasks[0].identifier == str(new_path)
    assert tasks[0].old_identifier == old_path_str


@pytest.mark.asyncio
async def test_catch_up_mtime_changed_and_content_changed_emits_upsert(watcher, manifest, corpus, ds_id):
    path = corpus / "doc.txt"
    path.write_bytes(b"version 2")
    stat = path.stat()
    # Seed the previous run's state: old content hash plus an mtime that differs
    # from the file's current mtime. Offsetting the stored mtime (rather than
    # rewriting the file and hoping mtime advances) keeps this deterministic on
    # filesystems like WSL2 where two rapid writes share an identical mtime.
    # mtime fast-path fails -> hash comparison fires -> upsert.
    seed_file(manifest, path, ds_id, content=b"version 1", mtime=stat.st_mtime - 1.0)
    queue = asyncio.Queue()
    await watcher._catch_up(queue, manifest)
    tasks = drain(queue)
    assert len(tasks) == 1
    assert tasks[0].change_type == "upsert"


@pytest.mark.asyncio
async def test_catch_up_mtime_changed_but_content_same_emits_no_task(watcher, manifest, corpus, ds_id):
    path = corpus / "doc.txt"
    path.write_bytes(b"content")
    stat = path.stat()
    # Stored mtime is slightly off (simulates a "touch"), but stored hash matches
    # current content — mtime check fails, falls through to hash, hash matches, no task.
    seed_file(manifest, path, ds_id, content=b"content", mtime=stat.st_mtime - 1.0)
    queue = asyncio.Queue()
    await watcher._catch_up(queue, manifest)
    assert drain(queue) == []


@pytest.mark.asyncio
async def test_catch_up_mtime_absent_falls_back_to_hash_and_emits_upsert(watcher, manifest, corpus, ds_id):
    path = corpus / "doc.txt"
    path.write_bytes(b"new content")
    # Seed with stale hash and no mtime entry — mtime fast-path is skipped,
    # hash comparison fires and detects the change.
    seed_file_no_mtime(manifest, path, ds_id, content=b"old content")
    queue = asyncio.Queue()
    await watcher._catch_up(queue, manifest)
    tasks = drain(queue)
    assert len(tasks) == 1
    assert tasks[0].change_type == "upsert"


@pytest.mark.asyncio
async def test_catch_up_deleted_file_emits_delete(watcher, manifest, corpus, ds_id):
    ghost_path = str(corpus / "ghost.txt")
    file_id = hash_source_ref(ghost_path)
    with manifest.transaction():
        manifest._conn.execute(
            "INSERT OR REPLACE INTO files "
            "(id, source_ref, content_hash, data_source_id, file_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (file_id, ghost_path, blake2b(b"old"), ds_id, "txt"),
        )
        # Inode chosen to be unreachable by any real file in the temp corpus.
        manifest.upsert_file_metadata(file_id, {"inode": 999_999_999, "mtime": 1000.0})

    queue = asyncio.Queue()
    await watcher._catch_up(queue, manifest)
    tasks = drain(queue)
    assert len(tasks) == 1
    assert tasks[0].change_type == "delete"
    assert tasks[0].identifier == ghost_path


# ---------------------------------------------------------------------------
# _catch_up — skipped files and logging
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_catch_up_unsupported_file_not_queued_and_logged(watcher, manifest, corpus, caplog):
    (corpus / "image.png").write_bytes(b"fake image data")
    queue = asyncio.Queue()
    with caplog.at_level(logging.INFO, logger="spruceup.monitoring.local_file_watcher"):
        await watcher._catch_up(queue, manifest)
    assert drain(queue) == []
    assert "1 skipped" in caplog.text
    assert "documentation" in caplog.text


@pytest.mark.asyncio
async def test_catch_up_directory_not_counted_as_skipped(watcher, manifest, corpus, caplog):
    (corpus / "subdir").mkdir()
    queue = asyncio.Queue()
    with caplog.at_level(logging.INFO, logger="spruceup.monitoring.local_file_watcher"):
        await watcher._catch_up(queue, manifest)
    assert drain(queue) == []
    assert "0 skipped" in caplog.text
    assert "documentation" not in caplog.text


# ---------------------------------------------------------------------------
# _watch — awatch patched with finite async generators
# ---------------------------------------------------------------------------

def make_awatch(*batches: list[tuple]):
    """Return a mock awatch that yields the given change batches then stops."""
    async def _mock(path, **kwargs):
        for batch in batches:
            yield set(batch)
    return _mock


@pytest.mark.asyncio
async def test_watch_added_supported_file_emits_upsert(watcher, manifest, corpus):
    path = corpus / "new.txt"
    path.write_bytes(b"content")
    catchup_done = asyncio.Event()
    catchup_done.set()
    queue = asyncio.Queue()
    with patch("spruceup.monitoring.local_file_watcher.awatch", make_awatch([(Change.added, str(path))])):
        await watcher._watch(queue, manifest, catchup_done)
    tasks = drain(queue)
    assert len(tasks) == 1
    assert tasks[0].change_type == "upsert"
    assert tasks[0].identifier == str(path)


@pytest.mark.asyncio
async def test_watch_modified_file_emits_upsert(watcher, manifest, corpus):
    path = corpus / "doc.txt"
    path.write_bytes(b"content")
    catchup_done = asyncio.Event()
    catchup_done.set()
    queue = asyncio.Queue()
    with patch("spruceup.monitoring.local_file_watcher.awatch", make_awatch([(Change.modified, str(path))])):
        await watcher._watch(queue, manifest, catchup_done)
    tasks = drain(queue)
    assert len(tasks) == 1
    assert tasks[0].change_type == "upsert"


@pytest.mark.asyncio
async def test_watch_deleted_file_emits_delete(watcher, manifest, corpus):
    # The file doesn't exist on disk (it was deleted); the watcher just emits the delete task.
    path_str = str(corpus / "gone.txt")
    catchup_done = asyncio.Event()
    catchup_done.set()
    queue = asyncio.Queue()
    with patch("spruceup.monitoring.local_file_watcher.awatch", make_awatch([(Change.deleted, path_str)])):
        await watcher._watch(queue, manifest, catchup_done)
    tasks = drain(queue)
    assert len(tasks) == 1
    assert tasks[0].change_type == "delete"
    assert tasks[0].identifier == path_str


@pytest.mark.asyncio
async def test_watch_move_detected_by_inode_emits_move(watcher, manifest, corpus, ds_id):
    new_path = corpus / "new_name.txt"
    new_path.write_bytes(b"content")
    old_path_str = str(corpus / "old_name.txt")

    # Seed the manifest with the old path, giving it the real file's inode so
    # the watcher can correlate the delete + add pair as a move.
    file_id = hash_source_ref(old_path_str)
    with manifest.transaction():
        manifest._conn.execute(
            "INSERT OR REPLACE INTO files "
            "(id, source_ref, content_hash, data_source_id, file_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (file_id, old_path_str, blake2b(b"content"), ds_id, "txt"),
        )
        manifest.upsert_file_metadata(file_id, {"inode": new_path.stat().st_ino})

    catchup_done = asyncio.Event()
    catchup_done.set()
    queue = asyncio.Queue()
    with patch(
        "spruceup.monitoring.local_file_watcher.awatch",
        make_awatch([(Change.deleted, old_path_str), (Change.added, str(new_path))]),
    ):
        await watcher._watch(queue, manifest, catchup_done)

    tasks = drain(queue)
    assert len(tasks) == 1
    assert tasks[0].change_type == "move"
    assert tasks[0].identifier == str(new_path)
    assert tasks[0].old_identifier == old_path_str


@pytest.mark.asyncio
async def test_watch_unsupported_added_file_not_queued(watcher, manifest, corpus):
    path = corpus / "image.png"
    path.write_bytes(b"fake image")
    catchup_done = asyncio.Event()
    catchup_done.set()
    queue = asyncio.Queue()
    with patch("spruceup.monitoring.local_file_watcher.awatch", make_awatch([(Change.added, str(path))])):
        await watcher._watch(queue, manifest, catchup_done)
    assert drain(queue) == []


@pytest.mark.asyncio
async def test_watch_added_nonexistent_path_not_queued(watcher, manifest, corpus):
    # Simulates a file that was added and immediately deleted before the event was processed.
    ghost = str(corpus / "ghost.txt")
    catchup_done = asyncio.Event()
    catchup_done.set()
    queue = asyncio.Queue()
    with patch("spruceup.monitoring.local_file_watcher.awatch", make_awatch([(Change.added, ghost)])):
        await watcher._watch(queue, manifest, catchup_done)
    assert drain(queue) == []


@pytest.mark.asyncio
async def test_watch_events_buffered_before_catchup_done(watcher, manifest, corpus):
    path = corpus / "doc.txt"
    path.write_bytes(b"content")
    catchup_done = asyncio.Event()  # never set
    queue = asyncio.Queue()
    with patch("spruceup.monitoring.local_file_watcher.awatch", make_awatch([(Change.added, str(path))])):
        await watcher._watch(queue, manifest, catchup_done)
    assert drain(queue) == []


@pytest.mark.asyncio
async def test_watch_buffered_events_flushed_when_catchup_done_set(watcher, manifest, corpus):
    file_a = corpus / "a.txt"
    file_b = corpus / "b.txt"
    file_a.write_bytes(b"a")
    file_b.write_bytes(b"b")

    catchup_done = asyncio.Event()
    queue = asyncio.Queue()

    async def mock_awatch(path, **kwargs):
        yield {(Change.added, str(file_a))}  # catchup_done not yet set → buffered
        catchup_done.set()
        yield {(Change.added, str(file_b))}  # now set → flush file_a and file_b together

    with patch("spruceup.monitoring.local_file_watcher.awatch", mock_awatch):
        await watcher._watch(queue, manifest, catchup_done)

    tasks = drain(queue)
    assert len(tasks) == 2
    identifiers = {t.identifier for t in tasks}
    assert str(file_a) in identifiers
    assert str(file_b) in identifiers
