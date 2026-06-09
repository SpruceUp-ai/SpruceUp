from spruceup.sync_engine import SyncEngine

from fakes import FakeTarget, make_chunk, make_file


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
