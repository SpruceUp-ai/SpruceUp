import logging

from .models import ChunkWrapper
from .models import SyncTask
from .utils.hashing import hash_chunk_id, hash_chunk_content
from .sync_engine import SyncEngine
from .utils.validation import validate_schema_objects
from .connectors.base import EmbedderConnector 
import asyncio

log = logging.getLogger(__name__)

class Coordinator:
    """Long-lived service that pulls SyncTasks from the queue and drives the pipeline."""

    def __init__(
        self,
        queue: object,
        transform,
        embedder: EmbedderConnector,
        sync_engine: SyncEngine,
        source_registry: dict,
    ):
        self._queue = queue
        self._transform = transform
        self._embedder = embedder
        self._sync_engine = sync_engine
        self._target = sync_engine._target
        self._manifest = sync_engine._manifest
        self._source_registry = source_registry
        self._active_tasks = set()

    async def process_task(self, task: SyncTask) -> None:
        source = self._source_registry[task.data_source_id]
        filename = source.display_name(task.identifier)
        try:
            if task.change_type == "delete":
                log.info("[delete] %s", filename)
                await self._sync_engine.delete_file(task.identifier)
            elif task.change_type == "move":
                old_name = source.display_name(task.old_identifier)
                log.info("[move] %s → %s", old_name, filename)
                await self._sync_engine.move_file(task.old_identifier, task.identifier)
            elif task.change_type == "upsert":
                log.info("[upsert] %s — transforming …", filename)
                await self.upsert_file(task, filename, source)
        except Exception:
            log.exception("[error] %s — task failed", filename)

    async def upsert_file(self, task: SyncTask, filename: str, source) -> None:
        from .memoize.context import (
            _memo_manifest_var, _memo_file_id_var, _memo_temp_keys_var, _memo_conn_var,
            _memo_stats_var,
        )

        spruce_file = await source.fetch(task)

        memo_conn = self._manifest.connect()
        try:
            with self._manifest.connect() as conn:
                self._manifest.ensure_file_row_exists(conn, spruce_file.file_id, spruce_file.source_ref)

            temp_keys: set[tuple[bytes, bytes]] = set()
            memo_stats = [0, 0]  # [hits, total]
            _memo_manifest_var.set(self._manifest)
            _memo_file_id_var.set(spruce_file.file_id)
            _memo_temp_keys_var.set(temp_keys)
            _memo_conn_var.set(memo_conn)
            _memo_stats_var.set(memo_stats)

            async def _embed(chunks):
                # Commit memoize writes accumulated before this yield point so
                # the write lock is released while we wait for the embedding API.
                # Other concurrent file tasks can then write to the manifest
                # without hitting "database is locked".
                memo_conn.commit()
                return await self._embedder.process_chunks(chunks)

            schema_objs = await self._transform(
                file_props={
                    "raw_content": source.decode_content(spruce_file.raw_content),
                    "source_ref": spruce_file.source_ref,
                    "modified_at": spruce_file.source_metadata.get("modified_at"),
                    "file_type": spruce_file.file_type,
                },
                embed=_embed,
            )
            memo_conn.commit()  # cover any writes that happen after embed returns
        finally:
            memo_conn.close()
            _memo_conn_var.set(None)

        if memo_stats[1] > 0:
            log.info("[memoize] %s — %d/%d hits", filename, memo_stats[0], memo_stats[1])

        self._manifest.sweep_memoized(spruce_file.file_id, temp_keys)

        validate_schema_objects(schema_objs, self._target.schema, self._target.primary_key)
        log.info("[upsert] %s — %d chunk(s)", filename, len(schema_objs))

        chunks = [
            ChunkWrapper(
                user_chunk=obj,
                user_chunk_object_hash=hash_chunk_content(obj),
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
            # Queue.get() only suspends when the queue is empty, so during catch-up
            # all N tasks are created before any run.  Yielding here lets each task
            # start (and the embedding batcher's flusher fire) before the next task
            # is created, so batches roll rather than all files piling up at once.
            await asyncio.sleep(0)
