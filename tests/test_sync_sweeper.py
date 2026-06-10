from spruceup.debounce_queue import DebounceQueue
from spruceup.sync_sweeper import SyncSweeper

from fakes import make_file


async def test_failed_delete_requeued_even_when_source_removed(manifest):
    source_id = manifest.register_source("local", "src")
    file = make_file(file_id="1:doc.txt", data_source_id=source_id)
    manifest.upsert_file_row(file)
    manifest.mark_failed(file.file_id, "delete")

    queue = DebounceQueue()
    sweeper = SyncSweeper(queue=queue, manifest=manifest, source_registry={})

    await sweeper._requeue_failed()

    tasks = []
    while not queue.empty():
        tasks.append(queue.get_nowait())

    assert len(tasks) == 1
    assert tasks[0].change_type == "delete"
    assert tasks[0].current_file_id == file.file_id
    assert tasks[0].data_source_id == source_id

    assert manifest.get_file_modified_at(file.file_id) is not None


async def test_failed_upsert_pruned_when_source_removed(manifest):
    source_id = manifest.register_source("local", "src")
    file = make_file(file_id="1:doc.txt", data_source_id=source_id)
    manifest.upsert_file_row(file)
    manifest.mark_failed(file.file_id, "upsert")

    queue = DebounceQueue()
    sweeper = SyncSweeper(queue=queue, manifest=manifest, source_registry={})

    await sweeper._requeue_failed()

    assert queue.empty()
    assert manifest.get_file_modified_at(file.file_id) is None


async def test_failed_upsert_requeued_when_source_present(manifest):
    source_id = manifest.register_source("local", "src")
    file = make_file(file_id="1:doc.txt", data_source_id=source_id)
    manifest.upsert_file_row(file)
    manifest.mark_failed(file.file_id, "upsert")

    queue = DebounceQueue()
    sweeper = SyncSweeper(
        queue=queue, manifest=manifest, source_registry={source_id: object()}
    )

    await sweeper._requeue_failed()

    assert not queue.empty()

    tasks = []
    while not queue.empty():
        tasks.append(queue.get_nowait())

    assert len(tasks) == 1
    assert tasks[0].change_type == "upsert"
    assert tasks[0].current_file_id == file.file_id
    assert tasks[0].data_source_id == source_id

    assert manifest.get_file_modified_at(file.file_id) is not None
