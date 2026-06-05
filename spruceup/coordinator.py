import asyncio
import logging

from .connectors.base import EmbedderConnector, TargetConnector
from .manifest import Manifest
from .models import ChunkWrapper, SyncTask
from .sync_engine import SyncEngine
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
        model_changed: bool = False,
    ):
        self._queue = queue
        self._transform = transform
        self._embedder = embedder
        self._sync_engine = sync_engine
        self._manifest = manifest
        self._target = target
        self._source_registry = source_registry
        self._model_changed = model_changed
        self._active_tasks = set()
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def process_task(self, task: SyncTask) -> None:
        source = self._source_registry[task.data_source_id]
        if task.change_type == "delete":
            filename = source.display_name(source.identifier_from_file_id(task.current_file_id))
            try:
                log.info("[delete] %s", filename)
                await self._sync_engine.delete_file(task.current_file_id)
            except Exception:
                log.exception("[error] %s — delete failed", filename)
                self._manifest.mark_failed(task.current_file_id, task.change_type)
        elif task.change_type == "upsert":
            filename = source.display_name(source.identifier_from_file_id(task.current_file_id))
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

        # Phase 1: fetch
        try:
            spruce_file = await source.fetch(task, self._manifest)
        except Exception:
            log.exception("[error] %s — fetch failed", filename)
            self._manifest.mark_failed(task.current_file_id, task.change_type)
            return

        spruce_file.force_upsert = self._model_changed
        self._manifest.ensure_file_row_exists(spruce_file.file_id)

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

        # Phase 2: transform (includes embed)
        try:
            user_chunks = await self._transform(
                file_props=FileProps(
                    raw_content=source.decode_content(spruce_file.raw_content),
                    display_name=spruce_file.display_name,
                    modified_at=spruce_file.modified_at,
                    file_type=spruce_file.file_type,
                ),
                embed=self._embedder.process_chunks,
            )
        except EmbeddingError:
            log.exception("[error] %s — embedding failed", filename)
            self._manifest.mark_failed(spruce_file.file_id, task.change_type)
            return
        except Exception:
            log.exception("[error] %s — transform failed", filename)
            self._manifest.mark_failed(spruce_file.file_id, task.change_type)
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

        # Phase 3: reconcile
        try:
            await self._sync_engine.reconcile(spruce_file)
        except Exception:
            log.exception("[error] %s — reconcile failed", filename)
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
