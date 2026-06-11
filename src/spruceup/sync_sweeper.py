import asyncio
import logging

from .debounce_queue import DebounceQueue
from .models import SyncTask

log = logging.getLogger(__name__)


class SyncSweeper:
    def __init__(
        self,
        queue: DebounceQueue,
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
        records = self._manifest.get_files_to_retry()
        requeued = 0
        for rec in records:
            file_id = rec["file_id"]
            ds_id = rec["data_source_id"]
            change_type = rec["change_type"]
            source = self._source_registry.get(ds_id)
            if source is not None or change_type == "delete":
                await self._queue.put(
                    SyncTask(
                        change_type,
                        current_file_id=file_id,
                        data_source_id=ds_id,
                    )
                )
                requeued += 1
            if source is None and change_type == "upsert":
                self._manifest.delete_file_row(file_id)

        if requeued:
            log.info("Sync sweeper — re-enqueuing %d failed file(s)", requeued)
