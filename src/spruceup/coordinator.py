import asyncio
import logging

from .connectors.base import EmbedderConnector, EmbeddingError, TargetConnector
from .manifest import Manifest
from .models import ChunkWrapper, FileProps, SyncTask
from .sync_engine import SyncEngine
from .transform_context import TransformContext, _transform_context
from .utils.hashing import hash_chunk_content
from .utils.validation import validate_schema_objects

log = logging.getLogger(__name__)

class Coordinator:
    def __init__(
        self,
        queue: object,
        transform,
        embedder: EmbedderConnector,
        sync_engine: SyncEngine,
        manifest: Manifest,
        target: TargetConnector,
        source_registry: dict,
        max_concurrency: int = 32,
    ):
        self._queue = queue
        self._transform = transform
        self._embedder = embedder
        self._sync_engine = sync_engine
        self._manifest = manifest
        self._target = target
        self._source_registry = source_registry
        self._active_tasks = set()
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def process_task(self, task: SyncTask) -> None:
        if task.change_type == "delete":
            # A delete only needs the file_id — no source required (its source
            # may even have been removed from the config).
            try:
                log.info("[delete] %s", task.current_file_id)
                await self._sync_engine.delete_file(task.current_file_id)
            except Exception:
                log.exception("[error] %s — delete failed", task.current_file_id)
                self._manifest.mark_failed(task.current_file_id, task.change_type)
        elif task.change_type == "upsert":
            # The source is needed only here, to fetch the file.
            source = self._source_registry[task.data_source_id]
            log.info("[upsert] %s — transforming …", task.current_file_id)
            await self.upsert_file(task, source)

    async def upsert_file(self, task: SyncTask, source) -> None:
        # file_id is the only label available until the fetch yields the real
        # display name.
        label = task.current_file_id

        # Phase 1: fetch
        try:
            spruce_file = await source.fetch(task, self._manifest)
        except Exception:
            log.exception("[error] %s — fetch failed", label)
            self._manifest.mark_failed(task.current_file_id, task.change_type)
            return

        label = spruce_file.display_name

        # Stale-task guard: a newer version of this file already synced. Abort
        # before touching the row or running transform/embed, leaving the newer
        # task's 'synced' state intact rather than marking this stale pass synced.
        # (reconcile re-checks to cover the concurrency window after this point.)
        stored_modified_at = self._manifest.get_file_modified_at(spruce_file.file_id)
        if stored_modified_at is not None and spruce_file.modified_at < stored_modified_at:
            log.debug("[stale] %s — skipped", label)
            return

        # Write the row now (in_flight) so it exists for the per-file cache and
        # chunk foreign keys; reconcile flips it to 'synced' once the target write
        # lands.
        self._manifest.upsert_file_row(spruce_file)

        # Per-file context the @memoize decorator and EmbeddingBatcher read while
        # the transform runs; they mutate its counters/used-key sets in place.
        ctx = TransformContext(manifest=self._manifest, file_id=spruce_file.file_id)
        _transform_context.set(ctx)

        # Phase 2: transform (includes embed)
        try:
            user_chunks = await self._transform(
                file_props=FileProps(
                    raw_content=source.decode_content(
                        spruce_file.raw_content, spruce_file.file_type
                    ),
                    display_name=spruce_file.display_name,
                    file_type=spruce_file.file_type,
                ),
                embed=self._embedder.process_chunks,
            )
        except EmbeddingError:
            log.exception("[error] %s — embedding failed", label)
            self._manifest.mark_failed(spruce_file.file_id, task.change_type)
            return
        except Exception:
            log.exception("[error] %s — transform failed", label)
            self._manifest.mark_failed(spruce_file.file_id, task.change_type)
            return

        if ctx.memo_total > 0:
            log.info("[memoize] %s — %d/%d hits", label, ctx.memo_hits, ctx.memo_total)
        if ctx.embed_total > 0:
            log.info("[embed_cache] %s — %d/%d hits", label, ctx.embed_hits, ctx.embed_total)

        self._manifest.sweep_memoized(spruce_file.file_id, ctx.memo_temp_keys)
        self._manifest.sweep_embedding_cache(spruce_file.file_id, ctx.embed_used_hashes)

        validate_schema_objects(user_chunks, self._target.schema)
        log.info("[upsert] %s — %d chunk(s)", label, len(user_chunks))

        chunks = [
            ChunkWrapper(
                user_chunk=obj,
                user_chunk_object_hash=hash_chunk_content(obj, self._target.vector_column),
            )
            for obj in user_chunks
        ]
        spruce_file.chunks = chunks

        # Phase 3: reconcile
        try:
            await self._sync_engine.reconcile(spruce_file)
        except Exception:
            log.exception("[error] %s — reconcile failed", label)
            self._manifest.mark_failed(spruce_file.file_id, task.change_type)
            return

        self._manifest.set_sync_state(spruce_file.file_id, "synced")

    async def run(self) -> None:
        while True:
            next_task: SyncTask = await self._queue.get()
            await self._semaphore.acquire()
            asyncio_task = asyncio.create_task(self._process_and_release(next_task))
            self._active_tasks.add(asyncio_task)
            asyncio_task.add_done_callback(self._active_tasks.discard)
            await asyncio.sleep(0)

    async def _process_and_release(self, task: SyncTask) -> None:
        try:
            await self.process_task(task)
        finally:
            self._semaphore.release()
            self._queue.task_done()
