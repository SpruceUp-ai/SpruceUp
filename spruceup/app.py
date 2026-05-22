import asyncio
import logging

import spruceup.registry as registry
from spruceup.coordinator import Coordinator
from spruceup.db import init_db
from spruceup.embedding import Embedder
from spruceup.manifest import Manifest
from spruceup.monitoring.monitor import Monitor
from spruceup.sync_engine import SyncEngine

log = logging.getLogger(__name__)

MANIFEST_PATH = "spruceup_manifest.db"


async def run(pipeline) -> None:
    init_db(MANIFEST_PATH)
    manifest = Manifest(MANIFEST_PATH)

    config = pipeline.config

    log.info(
        "SpruceUp starting — manifest=%s  target=%s",
        MANIFEST_PATH, config.target.table,
    )

    force_reindex = manifest.transform_hashes_changed(registry.tracker.hashes)
    if force_reindex:
        log.info("Transform functions changed — full reindex scheduled")
    else:
        log.info("Transform functions unchanged — incremental sync")

    sync_engine = SyncEngine(manifest=manifest, sync_target=config.target.create_sync_target())
    sync_engine.define_target_table(
        table_name=config.target.table,
        schema_from_class=config.target.schema,
        primary_key=config.target.primary_key,
    )

    embedder = Embedder(provider=config.embeddings.create_provider())
    queue: asyncio.Queue = asyncio.Queue()

    coordinator = Coordinator(
        queue=queue,
        transform=registry.transform_fn,
        embedder=embedder,
        sync_engine=sync_engine,
        schema_class=config.target.schema,
        primary_key=config.target.primary_key,
    )

    monitor = Monitor(queue, manifest, transform_tracker=registry.tracker)
    for source in config.sources:
        monitor.add_watcher(source.create_watcher())
    startup_done = asyncio.Event()

    monitor_task = asyncio.create_task(monitor.run(force_reindex, startup_done))
    coordinator_task = asyncio.create_task(coordinator.run())

    await startup_done.wait()
    watched = ", ".join(repr(source) for source in config.sources)
    log.info("Startup complete — watching %s for changes", watched)

    await asyncio.gather(monitor_task, coordinator_task)
