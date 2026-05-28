import asyncio
import logging

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
    force_reindex = manifest.transform_hash_changed(transform_hash)
    if force_reindex:
        log.info("Transform function changed — full reindex scheduled")
    else:
        log.info("Transform function unchanged — incremental sync")

    config.target.ensure_table_exists(
        embedding_dimensions=config.embedder.embedding_dimensions
    )
    sync_engine = SyncEngine(manifest=manifest, target=config.target)

    queue: asyncio.Queue = asyncio.Queue()

    monitor = Monitor(queue, manifest, transform_hash=transform_hash)
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
        source_registry=source_registry,
    )

    startup_done = asyncio.Event()

    monitor_task = asyncio.create_task(monitor.run(force_reindex, startup_done))
    coordinator_task = asyncio.create_task(coordinator.run())

    await startup_done.wait()
    watched = ", ".join(repr(source) for source in config.sources)
    log.info("Startup complete — watching %s for changes", watched)

    await asyncio.gather(monitor_task, coordinator_task)
