import asyncio
import logging
import time

from spruceup.connectors.embedders.embedding_batcher import EmbeddingBatcher
from spruceup.coordinator import Coordinator
from spruceup.debounce_queue import DebounceQueue
from spruceup.manifest import Manifest
from spruceup.memoize.decorator import _memoize_fn_hashes
from spruceup.models import SyncTask
from spruceup.monitoring.monitor import Monitor
from spruceup.sync_engine import SyncEngine
from spruceup.sync_sweeper import SyncSweeper
from spruceup.utils.hashing import hash_schema, hash_transform

log = logging.getLogger(__name__)


async def run(pipeline) -> None:
    manifest = Manifest()

    config = pipeline.config

    log.info(
        "SpruceUp starting — manifest=%s  target=%s",
        manifest.path, config.target.display_name,
    )

    transform_hash = hash_transform(config.transform)
    model = config.embedder.model
    dimensions = str(config.embedder.embedding_dimensions)
    target_identity = config.target.identity()
    schema_fingerprint = hash_schema(config.target.schema, config.target.vector_column)

    def _changed(key: str, current: str) -> bool:
        # First run (stored is None) is not a "change" — the value is just
        # recorded for next time. force_reindex is driven by the transform hash.
        stored = manifest.get_config_value(key)
        return stored is not None and stored != current

    transform_changed = manifest.transform_hash_changed(transform_hash)
    memoize_changed = manifest.any_memoize_fn_hash_missing(_memoize_fn_hashes)
    model_changed = _changed("embedding_model", model)
    dimensions_changed = _changed("embedding_dimensions", dimensions)
    target_changed = _changed("target_identity", target_identity)
    schema_changed = _changed("schema_fingerprint", schema_fingerprint)

    # Re-embedding is only required when the vectors themselves are invalidated.
    embeddings_invalidated = model_changed or dimensions_changed
    # The table/index must be dropped and rebuilt when its shape or destination
    # changes; a transform/memoize change reuses the existing structure.
    structure_changed = dimensions_changed or target_changed or schema_changed

    force_reindex = (
        transform_changed or memoize_changed or model_changed
        or dimensions_changed or target_changed or schema_changed
    )
    if force_reindex:
        reasons = []
        if transform_changed:
            reasons.append("transform function changed")
        if memoize_changed:
            reasons.append("memoized function changed")
        if model_changed:
            reasons.append("embedding model changed")
        if dimensions_changed:
            reasons.append("embedding dimensions changed")
        if target_changed:
            reasons.append("target changed")
        if schema_changed:
            reasons.append("schema changed")
        if embeddings_invalidated:
            manifest.flush_embedding_cache()
        log.info("Full reindex scheduled — %s", ", ".join(reasons))
    else:
        log.info("No changes detected — incremental sync")

    def persist_config_state() -> None:
        manifest.set_config_value("embedding_model", model)
        manifest.set_config_value("embedding_dimensions", dimensions)
        manifest.set_config_value("target_identity", target_identity)
        manifest.set_config_value("schema_fingerprint", schema_fingerprint)

    source_types: dict[type, list] = {}
    for source in config.sources:
        source_types.setdefault(type(source), []).append(source)
    for source_cls, typed_sources in source_types.items():
        await source_cls.validate(typed_sources)

    config.target.ensure_table_exists(
        embedding_dimensions=config.embedder.embedding_dimensions,
        recreate=structure_changed,
    )
    if structure_changed:
        log.info("Target %s rebuilt (drop + recreate)", config.target.display_name)
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

        embedder = EmbeddingBatcher(config.embedder)

        coordinator = Coordinator(
            queue=queue,
            transform=config.transform,
            embedder=embedder,
            sync_engine=sync_engine,
            manifest=manifest,
            target=config.target,
            source_registry=source_registry,
            force_upsert=force_reindex,
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

        # Files whose source was removed from the config are deleted as ordinary
        # delete tasks — same path as any delete, so the sweeper retries
        # failures. Enqueued after the coordinator starts so a large backlog
        # drains instead of filling the bounded queue. Empty source rows left by
        # prior completed removals are purged lazily here.
        manifest.purge_empty_inactive_sources(active_source_ids)
        for rec in manifest.get_orphaned_files(active_source_ids):
            await queue.put(SyncTask(
                "", "delete", time.time(),
                current_file_id=rec["file_id"],
                data_source_id=rec["data_source_id"],
            ))

        await startup_done.wait()
        watched = ", ".join(repr(source) for source in config.sources)
        log.info("Startup complete — watching %s for changes", watched)

        try:
            if force_reindex:
                await queue.join()
                manifest.set_config_value("file_cache_ready", "true")
                manifest.update_transform_hash(transform_hash)
                manifest.update_memoize_fn_hashes(_memoize_fn_hashes)
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
            # Persisted after any pending reingest completes, so a crash mid-
            # reindex re-triggers it; also backfills keys absent from an older
            # manifest so future changes to them are detected.
            persist_config_state()
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
