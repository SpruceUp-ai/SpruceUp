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
        self._failed_files: list[str] = []
        self._semaphore = asyncio.Semaphore(max_concurrency)

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
            self._failed_files.append(filename)

    async def upsert_file(self, task: SyncTask, filename: str, source) -> None:
        from .memoize.context import (
            _memo_manifest_var, _memo_file_id_var, _memo_temp_keys_var, _memo_stats_var,
            _embed_text_hashes_var,
        )

        spruce_file = await source.fetch(task)

        with self._manifest.connect() as conn:
            self._manifest.ensure_file_row_exists(conn, spruce_file.file_id, spruce_file.source_ref)

        temp_keys: set[tuple[bytes, bytes]] = set()
        memo_stats = [0, 0]  # [hits, total]
        # CachingEmbedder appends the text-hashes it computes here, in embed-call
        # order; we map them 1:1 onto the returned chunks below.
        embed_text_hashes: list[bytes] = []
        _memo_manifest_var.set(self._manifest)
        _memo_file_id_var.set(spruce_file.file_id)
        _memo_temp_keys_var.set(temp_keys)
        _memo_stats_var.set(memo_stats)
        _embed_text_hashes_var.set(embed_text_hashes)

        # No per-file memo connection / commit-before-yield wrapper: the manifest
        # uses one shared connection and set_memoized commits synchronously, so no
        # write transaction is ever held across the embed await.
        from .models import FileProps
        schema_objs = await self._transform(
            file_props=FileProps(
                raw_content=source.decode_content(spruce_file.raw_content),
                source_ref=spruce_file.source_ref,
                display_name=spruce_file.display_name,
                modified_at=spruce_file.source_metadata.get("modified_at"),
                file_type=spruce_file.file_type,
            ),
            embed=self._embedder.process_chunks,
        )

        if memo_stats[1] > 0:
            log.info("[memoize] %s — %d/%d hits", filename, memo_stats[0], memo_stats[1])

        self._manifest.sweep_memoized(spruce_file.file_id, temp_keys)

        validate_schema_objects(schema_objs, self._target.schema, self._target.primary_key)
        log.info("[upsert] %s — %d chunk(s)", filename, len(schema_objs))

        # Map embed text-hashes 1:1 onto chunks.
        # If the counts don't line up — transform made no embed() call, or violated the
        # 1:1 contract — drop provenance to `None rather than mis-attribute a hash
        # to the wrong chunk. `None` text_hash just means "no cache row to sweep
        # against," never a wrong hit.
        if len(embed_text_hashes) != len(schema_objs):
            if embed_text_hashes:
                log.warning(
                    "[cache] %s — %d embed hash(es) for %d chunk(s); "
                    "skipping text_hash provenance for this file",
                    filename, len(embed_text_hashes), len(schema_objs),
                )
            embed_text_hashes = [None] * len(schema_objs)

        chunks = [
            ChunkWrapper(
                user_chunk=obj,
                user_chunk_object_hash=hash_chunk_content(obj),
                ordinal=i,
                chunk_id=hash_chunk_id(task.identifier, i),
                text_hash=embed_text_hashes[i],
            )
            for i, obj in enumerate(schema_objs)
        ]

        spruce_file.chunks = chunks

        await self._sync_engine.reconcile([spruce_file])

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
