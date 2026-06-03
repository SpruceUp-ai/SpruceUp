import asyncio
import logging

from .models import SyncTask

log = logging.getLogger(__name__)


class SyncValidator:
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
            await self._requeue_failed()

    async def _requeue_failed(self) -> None:
        use_manifest_cache = self._manifest.get_config_value("file_cache_ready") == "true"
        total = 0
        for data_source_id, source in self._source_registry.items():
            for source_ref, ds_id in self._manifest.get_failed_files(data_source_id):
                await self._queue.put(SyncTask(
                    source.source_type,
                    source_ref,
                    "upsert",
                    data_source_id=ds_id,
                    use_manifest_cache=use_manifest_cache,
                ))
                total += 1
        if total:
            log.info("Sync validator — re-enqueuing %d failed file(s)", total)
