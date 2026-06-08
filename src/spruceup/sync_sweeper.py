import asyncio
import logging

from .models import SyncTask

log = logging.getLogger(__name__)


class SyncSweeper:
    def __init__(
        self,
        queue: asyncio.Queue,
        manifest,
        source_registry: dict,
        interval: float = 60.0,
    ):
        self._queue = queue
        self._manifest = manifest
        self._source_registry = source_registry
        self._interval = interval

    async def run(self) -> None:
        while True:
            await self._queue.join()
            await asyncio.sleep(self._interval)
            try:
                await self._requeue_failed()
            except Exception:
                log.exception("Sync sweeper error — will retry next interval")

    async def _requeue_failed(self) -> None:
        use_manifest_cache = self._manifest.get_config_value("file_cache_ready") == "true"
        records = self._manifest.get_failed_files()
        requeued = 0
        for rec in records:
            file_id = rec["file_id"]
            ds_id = rec["data_source_id"]
            change_type = rec["change_type"] or "upsert"
            source = self._source_registry.get(ds_id)
            if change_type == "delete":
                await self._queue.put(SyncTask(
                    "delete",
                    current_file_id=file_id,
                    data_source_id=ds_id,
                ))
                requeued += 1
            elif source is not None:
                await self._queue.put(SyncTask(
                    "upsert",
                    current_file_id=file_id,
                    data_source_id=ds_id,
                    use_manifest_cache=use_manifest_cache,
                ))
                requeued += 1
        if requeued:
            log.info("Sync sweeper — re-enqueuing %d failed file(s)", requeued)
