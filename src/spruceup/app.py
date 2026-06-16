import asyncio
import logging
from dataclasses import dataclass

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


@dataclass
class ReindexPlan:
    force_reindex: bool
    is_first_run: bool
    structure_changed: bool
    embeddings_invalidated: bool
    reasons: list[str]
    transform_hash: bytes
    fingerprints: dict[str, str]


def _plan_reindex(manifest: Manifest, config) -> ReindexPlan:
    first_run = manifest.is_first_run()

    transform_hash = hash_transform(config.transform)
    model = config.embedder.model
    dimensions = str(config.embedder.embedding_dimensions)
    target_identity = config.target.identity()
    schema_fingerprint = hash_schema(config.target.schema, config.target.vector_column)

    def _changed(key: str, current: str) -> bool:
        stored = manifest.get_config_value(key)
        return stored is not None and stored != current

    transform_changed = not first_run and manifest.transform_hash_changed(transform_hash)
    memoize_changed = not first_run and manifest.any_memoize_fn_hash_missing(_memoize_fn_hashes)
    model_changed = _changed("embedding_model", model)
    dimensions_changed = _changed("embedding_dimensions", dimensions)
    target_changed = _changed("target_identity", target_identity)
    schema_changed = _changed("schema_fingerprint", schema_fingerprint)

    reasons = [
        msg
        for changed, msg in (
            (transform_changed, "transform function changed"),
            (memoize_changed, "memoized function changed"),
            (model_changed, "embedding model changed"),
            (dimensions_changed, "embedding dimensions changed"),
            (target_changed, "target changed"),
            (schema_changed, "schema changed"),
        )
        if changed
    ]

    return ReindexPlan(
        force_reindex=first_run or bool(reasons),
        is_first_run=first_run,
        structure_changed=dimensions_changed or target_changed or schema_changed,
        embeddings_invalidated=model_changed or dimensions_changed,
        reasons=reasons,
        transform_hash=transform_hash,
        fingerprints={
            "embedding_model": model,
            "embedding_dimensions": dimensions,
            "target_identity": target_identity,
            "schema_fingerprint": schema_fingerprint,
        },
    )


async def run(pipeline) -> None:
    manifest = Manifest()

    config = pipeline.config

    log.info(
        "SpruceUp starting — manifest=%s  target=%s  file_cache=%s",
        manifest.path, config.target.display_name,
        "on" if config.cache_files else "off",
    )

    await config.embedder.health_check()
    

    plan = _plan_reindex(manifest, config)
    if plan.force_reindex:
        if plan.embeddings_invalidated:
            manifest.flush_embedding_cache()
        if plan.is_first_run:
            log.info("First startup — scheduling initial index")
        else:
            log.info("Full reindex scheduled — %s", ", ".join(plan.reasons))

    def persist_config_state() -> None:
        for key, value in plan.fingerprints.items():
            manifest.set_config_value(key, value)

    source_types: dict[type, list] = {}
    for source in config.sources:
        source_types.setdefault(type(source), []).append(source)
    for source_cls, typed_sources in source_types.items():
        await source_cls.validate(typed_sources)

    config.target.ensure_table_exists(
        embedding_dimensions=config.embedder.embedding_dimensions,
        recreate=plan.structure_changed,
    )
    if plan.structure_changed:
        log.info("Target %s rebuilt (drop + recreate)", config.target.display_name)
    embedder: EmbeddingBatcher | None = None
    try:
        sync_engine = SyncEngine(manifest=manifest, target=config.target)

        if plan.force_reindex:
            manifest.mark_all_files_needs_reindex()
        manifest.update_transform_hash(plan.transform_hash)
        manifest.update_memoize_fn_hashes(_memoize_fn_hashes)
        persist_config_state()

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

        manifest.purge_empty_inactive_sources(active_source_ids)
        for rec in manifest.get_orphaned_files(active_source_ids):
            await queue.put(SyncTask(
                "delete",
                current_file_id=rec["file_id"],
                data_source_id=rec["data_source_id"],
            ))
        for rec in manifest.get_needs_reindex_files(active_source_ids):
            await queue.put(SyncTask(
                rec["change_type"],
                current_file_id=rec["file_id"],
                data_source_id=rec["data_source_id"],
            ))

        embedder = EmbeddingBatcher(config.embedder)

        coordinator = Coordinator(
            queue=queue,
            transform=config.transform,
            embedder=embedder,
            sync_engine=sync_engine,
            manifest=manifest,
            target=config.target,
            source_registry=source_registry,
            cache_files=config.cache_files,
        )

        sync_sweeper = SyncSweeper(
            queue=queue,
            manifest=manifest,
            source_registry=source_registry,
        )

        startup_done = asyncio.Event()

        monitor_task = asyncio.create_task(monitor.run(startup_done))
        coordinator_task = asyncio.create_task(coordinator.run())
        sync_sweeper_task = asyncio.create_task(sync_sweeper.run())

        await startup_done.wait()
        coordinator.silent = False
        sync_engine.silent = False
        watched = ", ".join(repr(source) for source in config.sources)
        log.info("Watching %s for changes", watched)

        try:
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
