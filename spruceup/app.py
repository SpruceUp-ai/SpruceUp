import asyncio
import logging

from spruceup.connectors.embedders.caching import CachingEmbedder
from spruceup.connectors.embedders.embedding_batcher import EmbeddingBatcher
from spruceup.coordinator import Coordinator
from spruceup.manifest import Manifest
from spruceup.monitoring.monitor import Monitor
from spruceup.sync_engine import SyncEngine
from spruceup.utils.hashing import hash_transform

log = logging.getLogger(__name__)


async def run(pipeline) -> None:
    manifest = Manifest()

    config = pipeline.config

    log.info(
        "SpruceUp starting — manifest=%s  target=%s",
        manifest._path, config.target.display_name,
    )

    transform_hash = hash_transform(config.transform)
    transform_changed = manifest.transform_hash_changed(transform_hash)

    # The embedding spec ("model:dimensions") is a distinct reindex cause from a
    # transform change. A spec change means every cached vector lives in the
    # wrong space, so we wipe the cache UP FRONT — bound to this cause only. A
    # transform-only change must PRESERVE the cache (that preservation is what
    # makes a metadata-only edit a ~100% cache hit).
    embedding_spec = config.embedder.embedding_spec
    embedding_spec_changed = manifest.embedding_spec_changed(embedding_spec)
    if embedding_spec_changed:
        log.info("Embedding spec changed (%s) — wiping embedding cache", embedding_spec)
        manifest.wipe_embedding_cache()
    log.info("Embedding cache: %d row(s)", manifest.embedding_cache_size())

    force_reindex = transform_changed or embedding_spec_changed
    if force_reindex:
        causes = []
        if transform_changed:
            causes.append("transform changed")
        if embedding_spec_changed:
            causes.append("embedding spec changed")
        log.info("Full reindex scheduled (%s)", "; ".join(causes))
    else:
        log.info("Transform and embedding spec unchanged — incremental sync")

    source_types: dict[type, list] = {}
    for source in config.sources:
        source_types.setdefault(type(source), []).append(source)
    for source_cls, typed_sources in source_types.items():
        await source_cls.validate(typed_sources)

    config.target.ensure_table_exists(
        embedding_dimensions=config.embedder.embedding_dimensions
    )
    try:
        sync_engine = SyncEngine(manifest=manifest, target=config.target)

        queue: asyncio.Queue = asyncio.Queue()

        monitor = Monitor(queue, manifest)
        active_source_ids = []
        source_registry = {}
        for source in config.sources:
            data_source_id = manifest.register_source(source.source_type, source.source_identifier)
            active_source_ids.append(data_source_id)
            source_registry[data_source_id] = source
            monitor.add_watcher(source.create_watcher(data_source_id))
        await sync_engine.delete_stale_sources(active_source_ids)

        # Embedder chain (outermost first): cache → batcher → api. Caching is
        # outermost so hits are filtered before batching.
        embedder = CachingEmbedder(
            EmbeddingBatcher(config.embedder),
            manifest=manifest,
            embedding_spec=embedding_spec,
        )

        coordinator = Coordinator(
            queue=queue,
            transform=config.transform,
            embedder=embedder,
            sync_engine=sync_engine,
            source_registry=source_registry,
        )

        startup_done = asyncio.Event()

        monitor_task = asyncio.create_task(monitor.run(force_reindex, startup_done))
        coordinator_task = asyncio.create_task(coordinator.run())

        await startup_done.wait()
        watched = ", ".join(repr(source) for source in config.sources)
        log.info("Startup complete — watching %s for changes", watched)

        try:
            if force_reindex:
                await queue.join()
                if coordinator._failed_files:
                    log.warning(
                        "Reindex incomplete — %d file(s) failed; transform hash not "
                        "recorded, will retry next run. Failed: %s",
                        len(coordinator._failed_files),
                        ", ".join(coordinator._failed_files[:10]),
                    )
                else:
                    manifest.update_transform_hash(transform_hash)
                    manifest.update_embedding_spec(embedding_spec)
                    log.info("Reindex complete — transform hash + embedding spec recorded")
            await asyncio.gather(monitor_task, coordinator_task)
        except asyncio.CancelledError:
            monitor_task.cancel()
            coordinator_task.cancel()
            await asyncio.gather(monitor_task, coordinator_task, return_exceptions=True)
            raise
    finally:
        config.target.close()
        manifest._conn.close()
