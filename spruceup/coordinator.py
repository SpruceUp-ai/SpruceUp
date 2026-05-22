import logging

from .models import ChunkWrapper
from .monitoring.tasks import SyncTask
from .hashing import hash_chunk_id, hash_object
from .sync_engine import SyncEngine
from .validation import validate_schema_objects
import asyncio

log = logging.getLogger(__name__)


class Coordinator:
    """Long-lived service that pulls SyncTasks from the queue and drives the pipeline."""

    def __init__(
        self,
        queue: object,
        transform,
        embedder,
        sync_engine: SyncEngine,
        schema_class: type,
        primary_key: str,
        source_registry: dict,
    ):
        self._queue = queue
        self._transform = transform
        self._embedder = embedder
        self._sync_engine = sync_engine
        self._schema_class = schema_class
        self._primary_key = primary_key
        self._source_registry = source_registry
        self._active_tasks = set()

    async def process_task(self, task: SyncTask) -> None:
        source = self._source_registry[task.data_source_id]
        filename = source.display_name(task.identifier)
        try:
            await self._process_task(task, filename, source)
        except Exception:
            log.exception("[error] %s — task failed", filename)

    async def _process_task(self, task: SyncTask, filename: str, source) -> None:
        if task.change_type == "delete":
            log.info("[delete] %s", filename)
            self._sync_engine.delete_file(task.identifier)
            return

        if task.change_type == "move":
            old_name = source.display_name(task.old_identifier)
            log.info("[move] %s → %s", old_name, filename)
            self._sync_engine.move_file(task.old_identifier, task.identifier)
            return

        # upsert: full pipeline for this file
        spruce_file = await source.fetch(task)

        log.info("[upsert] %s — transforming …", filename)
        schema_objs = await self._transform(
            file_props={
                "raw_content": source.decode_content(spruce_file.raw_content),
                "file_path": spruce_file.file_path,
                "mtime": spruce_file.mtime,
                "file_type": spruce_file.file_type,
            },
            embed=self._embedder.process_chunks,
        )
        validate_schema_objects(schema_objs, self._schema_class, self._primary_key)
        log.info("[upsert] %s — %d chunk(s)", filename, len(schema_objs))

        chunks = [
            ChunkWrapper(
                user_chunk=obj,
                user_chunk_object_hash=hash_object(obj),
                ordinal=i,
                chunk_id=hash_chunk_id(task.identifier, i),
            )
            for i, obj in enumerate(schema_objs)
        ]

        spruce_file.chunks = chunks

        self._sync_engine.reconcile([spruce_file])

    async def run(self) -> None:
        while True:
            next_task: SyncTask = await self._queue.get()
            asyncio_task = asyncio.create_task(self.process_task(next_task))
            self._active_tasks.add(asyncio_task)
            asyncio_task.add_done_callback(self._active_tasks.discard)
