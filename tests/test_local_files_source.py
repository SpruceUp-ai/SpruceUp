"""
Tests for LocalFilesSource.

Uses real temp directories/files for stat()-dependent tests (inode, mtime).
No mocking of filesystem; asyncio.to_thread is not needed because LocalFilesSource
has no network I/O — only open() and os.stat(), which run on the main thread.
"""

import hashlib
import os
import pathlib

import pytest

from spruceup.connectors.sources.local import LocalFilesSource
from spruceup.models import SyncTask, SpruceFile
from spruceup.utils.hashing import hash_source_ref


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def blake2b(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=16).digest()


def make_task(path: str, data_source_id: int = 1) -> SyncTask:
    return SyncTask(
        source_type="local",
        identifier=path,
        change_type="upsert",
        data_source_id=data_source_id,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    return d


@pytest.fixture
def source(corpus):
    return LocalFilesSource(watched_dir=str(corpus))


# ---------------------------------------------------------------------------
# source_type
# ---------------------------------------------------------------------------

def test_source_type_is_local(source):
    assert source.source_type == "local"


# ---------------------------------------------------------------------------
# source_identifier
# ---------------------------------------------------------------------------

def test_source_identifier_resolves_to_absolute(tmp_path):
    src = LocalFilesSource(watched_dir=str(tmp_path / "corpus"))
    assert os.path.isabs(source_identifier := src.source_identifier)
    assert source_identifier == str(pathlib.Path(str(tmp_path / "corpus")).resolve())


def test_source_identifier_resolves_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rel_corpus").mkdir()
    src = LocalFilesSource(watched_dir="rel_corpus")
    assert src.source_identifier == str((tmp_path / "rel_corpus").resolve())


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_two_independent_dirs_passes(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    sources = [LocalFilesSource(str(a)), LocalFilesSource(str(b))]
    await LocalFilesSource.validate(sources)  # no exception


@pytest.mark.asyncio
async def test_validate_single_source_passes(tmp_path):
    d = tmp_path / "only"
    d.mkdir()
    await LocalFilesSource.validate([LocalFilesSource(str(d))])  # no exception


@pytest.mark.asyncio
async def test_validate_ancestor_descendant_raises(tmp_path):
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    with pytest.raises(ValueError, match="ancestor"):
        await LocalFilesSource.validate([
            LocalFilesSource(str(parent)),
            LocalFilesSource(str(child)),
        ])


@pytest.mark.asyncio
async def test_validate_error_message_names_both_paths(tmp_path):
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    with pytest.raises(ValueError) as exc_info:
        await LocalFilesSource.validate([
            LocalFilesSource(str(parent)),
            LocalFilesSource(str(child)),
        ])
    msg = str(exc_info.value)
    assert str(parent.resolve()) in msg
    assert str(child.resolve()) in msg


@pytest.mark.asyncio
async def test_validate_same_directory_raises(tmp_path):
    d = tmp_path / "shared"
    d.mkdir()
    with pytest.raises(ValueError):
        await LocalFilesSource.validate([
            LocalFilesSource(str(d)),
            LocalFilesSource(str(d)),
        ])


@pytest.mark.asyncio
async def test_validate_descendant_listed_first_still_raises(tmp_path):
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    with pytest.raises(ValueError, match="ancestor"):
        await LocalFilesSource.validate([
            LocalFilesSource(str(child)),
            LocalFilesSource(str(parent)),
        ])


# ---------------------------------------------------------------------------
# is_supported
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", [
    "doc.txt", "readme.md", "page.html", "data.json",
    "report.pdf", "letter.doc", "letter.docx",
])
def test_is_supported_returns_true_for_supported_extensions(source, filename):
    assert source.is_supported(filename) is True


@pytest.mark.parametrize("filename", [
    "photo.png", "image.jpg", "archive.zip", "script.py", "binary.exe",
])
def test_is_supported_returns_false_for_unsupported_extensions(source, filename):
    assert source.is_supported(filename) is False


def test_is_supported_case_insensitive(source):
    assert source.is_supported("readme.TXT") is True
    assert source.is_supported("README.MD") is True


def test_is_supported_no_extension_returns_false(source):
    assert source.is_supported("Makefile") is False


# ---------------------------------------------------------------------------
# create_watcher
# ---------------------------------------------------------------------------

def test_create_watcher_returns_local_file_watcher(source):
    from spruceup.monitoring.local_file_watcher import LocalFileWatcher
    watcher = source.create_watcher(data_source_id=42)
    assert isinstance(watcher, LocalFileWatcher)


# ---------------------------------------------------------------------------
# display_name
# ---------------------------------------------------------------------------

def test_display_name_returns_filename(source):
    assert source.display_name("/some/path/to/file.txt") == "file.txt"


def test_display_name_no_directory(source):
    assert source.display_name("standalone.md") == "standalone.md"


# ---------------------------------------------------------------------------
# decode_content
# ---------------------------------------------------------------------------

def test_decode_content_decodes_utf8(source):
    assert source.decode_content(b"hello world") == "hello world"


def test_decode_content_replaces_invalid_bytes(source):
    raw = b"valid \xff\xfe invalid"
    result = source.decode_content(raw)
    assert isinstance(result, str)
    assert "valid" in result


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_returns_spruce_file(source, corpus):
    path = corpus / "doc.txt"
    path.write_bytes(b"hello")
    task = make_task(str(path), data_source_id=7)
    result = await source.fetch(task)
    assert isinstance(result, SpruceFile)


@pytest.mark.asyncio
async def test_fetch_file_id_matches_hash_source_ref(source, corpus):
    path = corpus / "doc.txt"
    path.write_bytes(b"hello")
    result = await source.fetch(make_task(str(path)))
    assert result.file_id == hash_source_ref(str(path))


@pytest.mark.asyncio
async def test_fetch_source_ref_is_path_string(source, corpus):
    path = corpus / "doc.txt"
    path.write_bytes(b"hello")
    result = await source.fetch(make_task(str(path)))
    assert result.source_ref == str(path)


@pytest.mark.asyncio
async def test_fetch_display_name_is_filename(source, corpus):
    path = corpus / "report.pdf"
    path.write_bytes(b"%PDF")
    result = await source.fetch(make_task(str(path)))
    assert result.display_name == "report.pdf"


@pytest.mark.asyncio
async def test_fetch_content_hash_is_blake2b_of_bytes(source, corpus):
    content = b"some content"
    path = corpus / "file.txt"
    path.write_bytes(content)
    result = await source.fetch(make_task(str(path)))
    assert result.content_hash == blake2b(content)


@pytest.mark.asyncio
async def test_fetch_file_type_is_extension_without_dot(source, corpus):
    path = corpus / "notes.md"
    path.write_bytes(b"# Notes")
    result = await source.fetch(make_task(str(path)))
    assert result.file_type == "md"


@pytest.mark.asyncio
async def test_fetch_data_source_id_propagated_from_task(source, corpus):
    path = corpus / "doc.txt"
    path.write_bytes(b"x")
    result = await source.fetch(make_task(str(path), data_source_id=99))
    assert result.data_source_id == 99


@pytest.mark.asyncio
async def test_fetch_raw_content_matches_file_bytes(source, corpus):
    content = b"raw bytes here"
    path = corpus / "data.txt"
    path.write_bytes(content)
    result = await source.fetch(make_task(str(path)))
    assert result.raw_content == content


@pytest.mark.asyncio
async def test_fetch_chunks_starts_empty(source, corpus):
    path = corpus / "doc.txt"
    path.write_bytes(b"content")
    result = await source.fetch(make_task(str(path)))
    assert result.chunks == []


@pytest.mark.asyncio
async def test_fetch_source_metadata_contains_inode_and_mtime(source, corpus):
    path = corpus / "doc.txt"
    path.write_bytes(b"content")
    stat = path.stat()
    result = await source.fetch(make_task(str(path)))
    assert result.source_metadata["inode"] == stat.st_ino
    assert result.source_metadata["mtime"] == stat.st_mtime


@pytest.mark.asyncio
async def test_fetch_modified_at_equals_mtime(source, corpus):
    path = corpus / "doc.txt"
    path.write_bytes(b"content")
    result = await source.fetch(make_task(str(path)))
    assert result.source_metadata["modified_at"] == result.source_metadata["mtime"]
