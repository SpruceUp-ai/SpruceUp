import asyncio
import logging

from spruceup.connectors.embedders.embedding_batcher import EmbeddingBatcher
from spruceup.coordinator import Coordinator
from spruceup.debounce_queue import DebounceQueue
from spruceup.manifest import Manifest
from spruceup.memoize.decorator import _memoize_fn_hashes
from spruceup.monitoring.monitor import Monitor
from spruceup.sync_engine import SyncEngine
from spruceup.sync_sweeper import SyncSweeper
from spruceup.utils.hashing import hash_transform

log = logging.getLogger(__name__)


async def run(pipeline) -> None:
    manifest = Manifest()

    config = pipeline.config

    log.info(
        "SpruceUp starting — manifest=%s  target=%s",
        manifest.path, config.target.display_name,
    )

    transform_hash = hash_transform(config.transform)
    transform_changed = manifest.transform_hash_changed(transform_hash)
    memoize_changed = manifest.any_memoize_fn_hash_missing(_memoize_fn_hashes)
    stored_model = manifest.get_config_value("embedding_model")
    model_changed = stored_model is not None and stored_model != config.embedder.model
    force_reindex = transform_changed or memoize_changed or model_changed
    if force_reindex:
        reasons = []
        if transform_changed:
            reasons.append("transform function changed")
        if memoize_changed:
            reasons.append("memoized function changed")
        if model_changed:
            reasons.append("embedding model changed")
            manifest.flush_embedding_cache()
        log.info("Full reindex scheduled — %s", ", ".join(reasons))
    else:
        log.info("No changes detected — incremental sync")

    source_types: dict[type, list] = {}
    for source in config.sources:
        source_types.setdefault(type(source), []).append(source)
    for source_cls, typed_sources in source_types.items():
        await source_cls.validate(typed_sources)

    config.target.ensure_table_exists(
        embedding_dimensions=config.embedder.embedding_dimensions
    )
    embedder: EmbeddingBatcher | None = None
    try:
        sync_engine = SyncEngine(manifest=manifest, target=config.target)

        manifest.reset_in_flight_to_failed()

        queue: DebounceQueue = DebounceQueue()

        monitor = Monitor(queue, manifest)
        active_source_ids = []
        source_registry = {}
        for source in config.sources:
            data_source_id = manifest.register_source(source.source_type, source.source_identifier)
            active_source_ids.append(data_source_id)
            source_registry[data_source_id] = source
            monitor.add_watcher(source.create_watcher(data_source_id))
        await sync_engine.delete_stale_sources(active_source_ids)

        embedder = EmbeddingBatcher(config.embedder)

        coordinator = Coordinator(
            queue=queue,
            transform=config.transform,
            embedder=embedder,
            sync_engine=sync_engine,
            manifest=manifest,
            target=config.target,
            source_registry=source_registry,
            model_changed=model_changed,
        )

        sync_sweeper = SyncSweeper(
            queue=queue,
            manifest=manifest,
            source_registry=source_registry,
        )

        startup_done = asyncio.Event()

        monitor_task = asyncio.create_task(monitor.run(force_reindex, startup_done))
        coordinator_task = asyncio.create_task(coordinator.run())
        sync_sweeper_task = asyncio.create_task(sync_sweeper.run())

        await startup_done.wait()
        watched = ", ".join(repr(source) for source in config.sources)
        log.info("Startup complete — watching %s for changes", watched)

        try:
            if force_reindex:
                await queue.join()
                manifest.set_config_value("file_cache_ready", "true")
                manifest.update_transform_hash(transform_hash)
                manifest.update_memoize_fn_hashes(_memoize_fn_hashes)
                manifest.set_config_value("embedding_model", config.embedder.model)
                n_failed = len(manifest.get_failed_files())
                if n_failed:
                    log.warning(
                        "Reindex complete with %d failed file(s) — "
                        "sync sweeper will retry",
                        n_failed,
                    )
                else:
                    log.info("Reindex complete")
            elif manifest.get_config_value("file_cache_ready") is None:
                await queue.join()
                manifest.set_config_value("file_cache_ready", "true")
                log.info("Initial sync complete — file cache ready")
            await asyncio.gather(monitor_task, coordinator_task, sync_sweeper_task)
        except asyncio.CancelledError:
            monitor_task.cancel()
            coordinator_task.cancel()
            sync_sweeper_task.cancel()
            await asyncio.gather(
                monitor_task, coordinator_task, sync_sweeper_task,
                return_exceptions=True,
            )
            raise
    finally:
        await config.target.aclose()
        if embedder is not None:
            await embedder.aclose()
        manifest.close()
