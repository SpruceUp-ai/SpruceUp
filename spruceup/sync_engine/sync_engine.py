import logging
from typing import final

from ..connectors.base import TargetConnector
from ..manifest import Manifest
from ..models import ChunkWrapper, SpruceFile

log = logging.getLogger(__name__)


@final
class SyncEngine:
    def __init__(self, manifest: Manifest, target: TargetConnector) -> None:
        self._manifest = manifest
        self._target = target

    async def reconcile(self, file: SpruceFile) -> None:
        manifest_upserts: list[tuple[str, ChunkWrapper]] = []
        target_upserts: list[ChunkWrapper] = []
        manifest_deletes: list[tuple[str, bytes]] = []
        target_deletes: list[bytes] = []

        prev_chunks = self._manifest.get_chunks_for_file(file.file_id)
        prev_hashes: set[bytes] = {c["user_chunk_object_hash"] for c in prev_chunks}
        curr_hashes: dict[bytes, ChunkWrapper] = {
            chunk.user_chunk_object_hash: chunk for chunk in file.chunks
        }

        for h, chunk in curr_hashes.items():
            if file.force_upsert or h not in prev_hashes:
                manifest_upserts.append((file.file_id, chunk))
                target_upserts.append(chunk)

        for h in prev_hashes:
            if h not in curr_hashes:
                manifest_deletes.append((file.file_id, h))
                target_deletes.append(h)

        # The file row was already written (in_flight) by the coordinator before
        # transform, so it exists for the chunk foreign key here. The stale guard
        # ran there too, before that write.
        await self._target.sync(file.file_id, target_upserts, target_deletes)

        with self._manifest.transaction():
            self._manifest.upsert_chunks(manifest_upserts)
            self._manifest.delete_chunks(manifest_deletes)

        log.info(
            "Synced %s — %d upserted  %d deleted",
            file.display_name,
            len(target_upserts),
            len(target_deletes),
        )

    async def delete_file(self, file_id: str) -> None:
        chunks = self._manifest.get_chunks_for_file(file_id)
        hashes = [c["user_chunk_object_hash"] for c in chunks]
        await self._target.sync(file_id, [], hashes)
        self._manifest.delete_file_row(file_id)
        log.info("Deleted %d chunk(s) for file_id=%s", len(hashes), file_id)
