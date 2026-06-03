import logging

from .models import ChunkWrapper
from .models import SyncTask
from .utils.hashing import hash_chunk_content
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
        max_concurrency: int = 32,
    ):
        self._queue = queue
        self._transform = transform
        self._embedder = embedder
        self._sync_engine = sync_engine
        self._target = sync_engine._target
        self._manifest = sync_engine._manifest
        self._source_registry = source_registry
        self._active_tasks = set()
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def process_task(self, task: SyncTask) -> None:
        source = self._source_registry[task.data_source_id]
        filename = source.display_name(task.identifier)
        if task.change_type == "delete":
            try:
                log.info("[delete] %s", filename)
                await self._sync_engine.delete_file(task.identifier)
            except Exception:
                log.exception("[error] %s — delete failed", filename)
        elif task.change_type == "move":
            try:
                old_name = source.display_name(task.old_identifier)
                log.info("[move] %s → %s", old_name, filename)
                await self._sync_engine.move_file(task.old_identifier, task.identifier)
            except Exception:
                log.exception("[error] %s — move failed", filename)
        elif task.change_type == "upsert":
            log.info("[upsert] %s — transforming …", filename)
            await self.upsert_file(task, filename, source)

    async def upsert_file(self, task: SyncTask, filename: str, source) -> None:
        from .memoize.context import (
            _memo_manifest_var, _memo_file_id_var, _memo_temp_keys_var, _memo_stats_var,
        )
        from .connectors.embedders.context import (
            _embed_manifest_var, _embed_file_id_var, _embed_used_hashes_var, _embed_stats_var,
        )
        from .connectors.base import EmbeddingError
        from .models import FileProps

        # Phase 1: fetch — source boundary; watcher handles retries on failure
        try:
            spruce_file = await source.fetch(task, self._manifest)
        except Exception:
            log.exception("[error] %s — fetch failed", filename)
            return

        self._manifest.ensure_file_row_exists(spruce_file.file_id, spruce_file.source_ref)

        temp_keys: set[tuple[bytes, bytes]] = set()
        memo_stats = [0, 0]
        _memo_manifest_var.set(self._manifest)
        _memo_file_id_var.set(spruce_file.file_id)
        _memo_temp_keys_var.set(temp_keys)
        _memo_stats_var.set(memo_stats)

        embed_used_hashes: set[bytes] = set()
        embed_stats = [0, 0]
        _embed_manifest_var.set(self._manifest)
        _embed_file_id_var.set(spruce_file.file_id)
        _embed_used_hashes_var.set(embed_used_hashes)
        _embed_stats_var.set(embed_stats)

        # Phase 2: transform (includes embed) — EmbeddingError is caught and marked
        # failed; all other exceptions propagate and crash the app (user code bugs)
        try:
            user_chunks = await self._transform(
                file_props=FileProps(
                    raw_content=source.decode_content(spruce_file.raw_content),
                    source_ref=spruce_file.source_ref,
                    display_name=spruce_file.display_name,
                    modified_at=spruce_file.source_metadata.get("modified_at"),
                    file_type=spruce_file.file_type,
                ),
                embed=self._embedder.process_chunks,
            )
        except EmbeddingError:
            log.exception("[error] %s — embedding failed", filename)
            self._manifest.set_sync_state(spruce_file.file_id, "failed")
            return

        if memo_stats[1] > 0:
            log.info("[memoize] %s — %d/%d hits", filename, memo_stats[0], memo_stats[1])
        if embed_stats[1] > 0:
            log.info("[embed_cache] %s — %d/%d hits", filename, embed_stats[0], embed_stats[1])

        self._manifest.sweep_memoized(spruce_file.file_id, temp_keys)
        self._manifest.sweep_embedding_cache(spruce_file.file_id, embed_used_hashes)

        validate_schema_objects(user_chunks, self._target.schema)
        log.info("[upsert] %s — %d chunk(s)", filename, len(user_chunks))

        chunks = [
            ChunkWrapper(
                user_chunk=obj,
                user_chunk_object_hash=hash_chunk_content(obj),
            )
            for obj in user_chunks
        ]
        spruce_file.chunks = chunks

        # Phase 3: reconcile — target boundary; mark failed on error
        try:
            await self._sync_engine.reconcile(spruce_file)
        except Exception:
            log.exception("[error] %s — reconcile failed", filename)
            self._manifest.set_sync_state(spruce_file.file_id, "failed")
            return

        self._manifest.set_sync_state(spruce_file.file_id, "synced")

    async def run(self) -> None:
        while True:
            next_task: SyncTask = await self._queue.get()
            # Bound concurrent in-flight file tasks so peak memory (file bytes +
            # frames + embeddings) does not grow with corpus size.
            await self._semaphore.acquire()
            asyncio_task = asyncio.create_task(self._process_and_release(next_task))
            self._active_tasks.add(asyncio_task)
            asyncio_task.add_done_callback(self._active_tasks.discard)
            # Queue.get() only suspends when the queue is empty, so during catch-up
            # all N tasks are created before any run.  Yielding here lets each task
            # start (and the embedding batcher's flusher fire) before the next task
            # is created, so batches roll rather than all files piling up at once.
            await asyncio.sleep(0)

    async def _process_and_release(self, task: SyncTask) -> None:
        try:
            await self.process_task(task)
        finally:
            self._semaphore.release()
            self._queue.task_done()
