from unittest.mock import patch

from spruceup.coordinator import Coordinator
from spruceup.debounce_queue import DebounceQueue
from spruceup.models import SyncTask
from spruceup.sync_engine import SyncEngine

from fakes import FakeEmbedder, FakeSource, FakeTarget, make_file, make_transform


def build_coordinator(manifest, source_id, source, *, transform=None, embedder=None, target=None):
    embedder = embedder or FakeEmbedder()
    target = target or FakeTarget()
    engine = SyncEngine(manifest, target)
    coord = Coordinator(
        queue=DebounceQueue(),
        transform=transform or make_transform(2),
        embedder=embedder,
        sync_engine=engine,
        manifest=manifest,
        target=target,
        source_registry={source_id: source},
    )
    return coord, target, embedder


def sync_state(manifest, file_id):
    row = manifest._conn.execute(
        "SELECT sync_state FROM files WHERE id = ?", (file_id,)
    ).fetchone()
    return row[0] if row else None


async def test_process_task_upsert_happy_path(manifest):
    source_id = manifest.register_source("fake", "src")
    file = make_file(file_id="1:doc.txt", data_source_id=source_id)
    source = FakeSource(spruce_file=file)
    coord, target, embedder = build_coordinator(manifest, source_id, source)

    task = SyncTask("upsert", current_file_id=file.file_id, data_source_id=source_id)
    await coord.process_task(task)

    assert embedder.embedded_batches
    assert len(target.calls) == 1
    _, upserts, deletes = target.calls[0]
    assert len(upserts) == 2
    assert manifest.get_failed_files() == []
    assert sync_state(manifest, file.file_id) == "synced"


async def test_fetch_failure_records_retryable_failed_row(manifest):
    source_id = manifest.register_source("fake", "src")
    source = FakeSource(fetch_error=RuntimeError("boom"))
    coord, target, embedder = build_coordinator(manifest, source_id, source)

    task = SyncTask("upsert", current_file_id="1:doc.txt", data_source_id=source_id)
    await coord.process_task(task)

    assert target.calls == []
    assert embedder.embedded_batches == []

    failed = manifest.get_failed_files()
    assert len(failed) == 1
    assert failed[0]["file_id"] == "1:doc.txt"
    assert failed[0]["data_source_id"] == source_id
    assert failed[0]["change_type"] == "upsert"


async def test_transform_bug_marks_file_failed_without_crashing(manifest):
    source_id = manifest.register_source("fake", "src")
    file = make_file(file_id="1:doc.txt", data_source_id=source_id)
    source = FakeSource(spruce_file=file)

    async def boom(*, file_props, embed):
        raise RuntimeError("transform bug")

    coord, target, _ = build_coordinator(manifest, source_id, source, transform=boom)

    task = SyncTask("upsert", current_file_id=file.file_id, data_source_id=source_id)
    await coord.process_task(task)

    assert target.calls == []
    assert sync_state(manifest, file.file_id) == "failed"


async def test_stale_event_is_skipped_without_overriding_newer_data(manifest):
    source_id = manifest.register_source("fake", "src")
    newer = make_file(file_id="1:doc.txt", data_source_id=source_id, modified_at=200.0)
    manifest.upsert_file_row(newer)

    stale = make_file(file_id="1:doc.txt", data_source_id=source_id, modified_at=100.0)
    source = FakeSource(spruce_file=stale)
    coord, target, embedder = build_coordinator(manifest, source_id, source)

    task = SyncTask("upsert", current_file_id="1:doc.txt", data_source_id=source_id)
    await coord.process_task(task)

    assert target.calls == []
    assert embedder.embedded_batches == []
    assert manifest.get_file_modified_at("1:doc.txt") == 200.0


async def test_delete_task_routes_to_sync_engine(manifest):
    source_id = manifest.register_source("fake", "src")
    coord, _, _ = build_coordinator(manifest, source_id, FakeSource())

    task = SyncTask("delete", current_file_id="1:doc.txt", data_source_id=source_id)
    with (
        patch.object(
            coord._sync_engine, "delete_file", wraps=coord._sync_engine.delete_file
        ) as delete_file,
        patch.object(manifest, "mark_failed", wraps=manifest.mark_failed) as mark_failed,
    ):
        await coord.process_task(task)

    delete_file.assert_awaited_once_with("1:doc.txt")
    mark_failed.assert_not_called()


async def test_delete_failure_is_marked_failed(manifest):
    source_id = manifest.register_source("fake", "src")
    coord, _, _ = build_coordinator(manifest, source_id, FakeSource())

    task = SyncTask("delete", current_file_id="1:doc.txt", data_source_id=source_id)
    with (
        patch.object(coord._sync_engine, "delete_file", side_effect=RuntimeError("boom")),
        patch.object(manifest, "mark_failed", wraps=manifest.mark_failed) as mark_failed,
    ):
        await coord.process_task(task)

    mark_failed.assert_called_once_with("1:doc.txt", "delete")
