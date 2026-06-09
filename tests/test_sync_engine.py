from spruceup.sync_engine import SyncEngine

from fakes import FakeTarget, make_chunk, make_chunks, make_file


async def test_reconcile_dedupes_chunks_sharing_a_hash(manifest):
    source_id = manifest.register_source("local", "src")
    target = FakeTarget()
    engine = SyncEngine(manifest, target)

    h = b"shared-hash"
    file = make_file(data_source_id=source_id, chunks=[make_chunk(h), make_chunk(h)])
    manifest.upsert_file_row(file)

    await engine.reconcile(file)

    assert len(target.calls) == 1
    file_id, upserts, deletes = target.calls[0]
    assert file_id == file.file_id
    assert len(upserts) == 1
    assert deletes == []


async def test_reconcile_new_file_upserts_all_chunks(manifest):
    source_id = manifest.register_source("local", "src")
    target = FakeTarget()
    engine = SyncEngine(manifest, target)

    chunks = make_chunks(3)
    file = make_file(data_source_id=source_id, chunks=chunks)
    manifest.upsert_file_row(file)

    await engine.reconcile(file)

    assert len(target.calls) == 1
    _, upserts, deletes = target.calls[0]
    assert deletes == []
    assert len(upserts) == len(chunks)


async def test_reconcile_adds_and_removes_changed_chunks(manifest):
    source_id = manifest.register_source("local", "src")
    target = FakeTarget()
    engine = SyncEngine(manifest, target)

    a, b, c = make_chunk(b"a"), make_chunk(b"b"), make_chunk(b"c")
    file = make_file(data_source_id=source_id, chunks=[a, b])
    manifest.upsert_file_row(file)
    manifest.upsert_chunks([(file.file_id, a), (file.file_id, b)])

    file.chunks = [a, c]
    await engine.reconcile(file)

    assert len(target.calls) == 1
    _, upserts, deletes = target.calls[0]
    assert [chunk.user_chunk_object_hash for chunk in upserts] == [b"c"]
    assert deletes == [b"b"]


async def test_reconcile_force_upsert_repushes_already_synced_chunks(manifest):
    source_id = manifest.register_source("local", "src")
    target = FakeTarget()
    engine = SyncEngine(manifest, target, force_upsert=True)

    chunks = make_chunks(3)
    file = make_file(data_source_id=source_id, chunks=chunks)
    manifest.upsert_file_row(file)
    manifest.upsert_chunks([(file.file_id, c) for c in chunks])

    await engine.reconcile(file)

    assert len(target.calls) == 1
    _, upserts, deletes = target.calls[0]
    assert len(upserts) == len(chunks)
    assert deletes == []


async def test_delete_file_removes_chunks_from_target_and_manifest(manifest):
    source_id = manifest.register_source("local", "src")
    target = FakeTarget()
    engine = SyncEngine(manifest, target)

    chunks = make_chunks(2)
    file = make_file(data_source_id=source_id, chunks=chunks)
    manifest.upsert_file_row(file)
    manifest.upsert_chunks([(file.file_id, c) for c in chunks])

    await engine.delete_file(file.file_id)

    assert len(target.calls) == 1
    _, upserts, deletes = target.calls[0]
    assert upserts == []
    assert set(deletes) == {c.user_chunk_object_hash for c in chunks}

    assert manifest.get_chunks_for_file(file.file_id) == []
    assert manifest.get_file_modified_at(file.file_id) is None
