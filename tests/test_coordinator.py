import asyncio

import pytest

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


async def test_run_crashes_on_transform_bug(manifest):
    source_id = manifest.register_source("fake", "src")
    file = make_file(file_id="1:doc.txt", data_source_id=source_id)
    source = FakeSource(spruce_file=file)

    async def boom(*, file_props, embed):
        raise RuntimeError("transform bug")

    coord, _, _ = build_coordinator(manifest, source_id, source, transform=boom)
    coord._queue.put_nowait(
        SyncTask("upsert", current_file_id=file.file_id, data_source_id=source_id)
    )

    with pytest.raises(RuntimeError, match="transform bug"):
        await asyncio.wait_for(coord.run(), timeout=2.0)
