import asyncio
import logging

from .connectors.base import EmbedderConnector, EmbeddingError, TargetConnector
from .debounce_queue import DebounceQueue
from .manifest import Manifest
from .models import ChunkWrapper, FileProps, SyncTask
from .sync_engine import SyncEngine
from .transform_context import TransformContext, transform_scope
from .utils.hashing import hash_chunk_content
from .utils.validation import validate_schema_objects

log = logging.getLogger(__name__)

class Coordinator:
    def __init__(
        self,
        queue: DebounceQueue,
        transform,
        embedder: EmbedderConnector,
        sync_engine: SyncEngine,
        manifest: Manifest,
        target: TargetConnector,
        source_registry: dict,
        max_concurrency: int = 32,
        cache_files: bool = True,
    ):
        self._queue = queue
        self._transform = transform
        self._embedder = embedder
        self._sync_engine = sync_engine
        self._manifest = manifest
        self._target = target
        self._source_registry = source_registry
        self._cache_files = cache_files
        self._active_tasks = set()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._fatal: asyncio.Future[None] | None = None

    async def process_task(self, task: SyncTask) -> None:
        file_id = task.current_file_id
        if task.change_type == "delete":
            try:
                log.info("[delete] %s", file_id)
                await self._sync_engine.delete_file(file_id)
            except Exception:
                log.exception("[error] %s — delete failed", file_id)
                self._manifest.mark_failed(file_id, task.change_type)
        elif task.change_type == "upsert":
            source = self._source_registry[task.data_source_id]
            log.info("[upsert] %s — transforming …", file_id)
            await self.upsert_file(task, source)

    async def upsert_file(self, task: SyncTask, source) -> None:
        file_id = task.current_file_id
        label = file_id

        try:
            spruce_file = await source.fetch(task, self._manifest)
        except Exception:
            log.exception("[error] %s — fetch failed", label)
            self._manifest.mark_fetch_failed(file_id, task.data_source_id, task.change_type)
            return

        label = spruce_file.display_name

        stored_modified_at = self._manifest.get_file_modified_at(spruce_file.file_id)
        if stored_modified_at is not None and spruce_file.modified_at < stored_modified_at:
            log.debug("[stale] %s — skipped", label)
            return

        self._manifest.upsert_file_row(spruce_file, cache_content=self._cache_files)

        ctx = TransformContext(manifest=self._manifest, file_id=spruce_file.file_id)
        with transform_scope(ctx):
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
                # TODO: a deterministically failing file (e.g. corrupted content)
                # needs a dead-letter state; 'failed' means the sweeper retries
                # it every interval forever.
                log.exception(
                    "[error] %s — transform failed (possibly corrupted file)", label
                )
                self._manifest.mark_failed(spruce_file.file_id, task.change_type)
                return

        if ctx.memo_total > 0:
            log.info("[memoize] %s — %d/%d hits", label, ctx.memo_hits, ctx.memo_total)
        if ctx.embed_total > 0:
            log.info("[embed_cache] %s — %d/%d hits", label, ctx.embed_hits, ctx.embed_total)

        self._manifest.sweep_memoized(spruce_file.file_id, ctx.used_memoized_subfn_call_keys)
        self._manifest.sweep_embedding_cache(spruce_file.file_id, ctx.used_chunk_embedding_cache_keys)

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

        try:
            await self._sync_engine.reconcile(spruce_file)
        except Exception:
            log.exception("[error] %s — reconcile failed", label)
            self._manifest.mark_failed(spruce_file.file_id, task.change_type)
            return

        self._manifest.set_sync_state(spruce_file.file_id, "synced")

    async def run(self) -> None:
        self._fatal = asyncio.get_running_loop().create_future()
        while True:
            get_task = asyncio.ensure_future(self._queue.get())
            await asyncio.wait(
                {get_task, self._fatal}, return_when=asyncio.FIRST_COMPLETED
            )
            if self._fatal.done():
                get_task.cancel()
                await self._fatal  # re-raises a child task's fatal exception
            next_task: SyncTask = get_task.result()
            await self._semaphore.acquire()
            child = asyncio.create_task(self._process_and_release(next_task))
            self._active_tasks.add(child)
            child.add_done_callback(self._on_task_done)
            await asyncio.sleep(0)

    def _on_task_done(self, task: asyncio.Task) -> None:
        self._active_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None and self._fatal is not None and not self._fatal.done():
            self._fatal.set_exception(exc)

    async def _process_and_release(self, task: SyncTask) -> None:
        try:
            await self.process_task(task)
        finally:
            self._semaphore.release()
            self._queue.task_done()
