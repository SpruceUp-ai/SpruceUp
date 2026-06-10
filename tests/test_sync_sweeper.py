from conftest import manifest
from fakes import FakeTarget, make_chunk, make_chunks, make_file


def test_cleanup_orphaned_files_when_source_removed_after_upsert_failure(manifest):
    # register source
    # upsert a file
    # force failure of the upsert
    # delete source
    # assert that delete sync_task for file in debounce queue.
    pass
