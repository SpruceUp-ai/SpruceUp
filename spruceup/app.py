import asyncio
import logging

from spruceup.coordinator import Coordinator
from spruceup.embedding import Embedder
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
        MANIFEST_PATH, config.target.table,
    )

    transform_hash = hash_transform(config.transform)
    force_reindex = manifest.transform_hash_changed(transform_hash)
    if force_reindex:
        log.info("Transform function changed — full reindex scheduled")
    else:
        log.info("Transform function unchanged — incremental sync")

    sync_engine = SyncEngine(manifest=manifest, sync_target=config.target.create_sync_target())
    sync_engine.define_target_table(
        table_name=config.target.table,
        schema_from_class=config.target.schema,
        primary_key=config.target.primary_key,
    )

    embedder = Embedder(provider=config.embeddings.create_provider())
    queue: asyncio.Queue = asyncio.Queue()

    monitor = Monitor(queue, manifest, transform_hash=transform_hash)
    active_source_ids = []
    source_registry = {}
    for source in config.sources:
        data_source_id = manifest.register_source(source.source_type, source.source_identifier)
        active_source_ids.append(data_source_id)
        source_registry[data_source_id] = source
        monitor.add_watcher(source.create_watcher(data_source_id))
    manifest.delete_stale_sources(active_source_ids)

    coordinator = Coordinator(
        queue=queue,
        transform=config.transform,
        embedder=embedder,
        sync_engine=sync_engine,
        schema_class=config.target.schema,
        primary_key=config.target.primary_key,
        source_registry=source_registry,
    )

    startup_done = asyncio.Event()

    monitor_task = asyncio.create_task(monitor.run(force_reindex, startup_done))
    coordinator_task = asyncio.create_task(coordinator.run())

    await startup_done.wait()
    watched = ", ".join(repr(source) for source in config.sources)
    log.info("Startup complete — watching %s for changes", watched)

    await asyncio.gather(monitor_task, coordinator_task)
