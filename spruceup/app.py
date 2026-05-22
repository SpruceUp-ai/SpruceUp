import asyncio
import logging

import spruceup.registry as registry
from spruceup.coordinator import Coordinator
from spruceup.db import init_db
from spruceup.embedding import Embedder, OpenAIProvider
from spruceup.manifest import Manifest
from spruceup.monitoring.monitor import LocalFileWatcher, Monitor
from spruceup.sync_engine import SyncEngine

log = logging.getLogger(__name__)

MANIFEST_PATH = "spruceup_manifest.db"


async def run(pipeline) -> None:
    init_db(MANIFEST_PATH)
    manifest = Manifest(MANIFEST_PATH)

    log.info(
        "SpruceUp starting — manifest=%s  target=%s",
        MANIFEST_PATH, pipeline.TARGET_TABLE,
    )

    force_reindex = manifest.transform_hashes_changed(registry.tracker.hashes)
    if force_reindex:
        log.info("Transform functions changed — full reindex scheduled")
    else:
        log.info("Transform functions unchanged — incremental sync")

    sync_engine = SyncEngine(manifest=manifest, pg_connstr=pipeline.PG_CONNSTR)
    sync_engine.define_target_table(
        table_name=pipeline.TARGET_TABLE,
        schema_from_class=pipeline.CHUNK_SCHEMA,
        primary_key=pipeline.PRIMARY_KEY,
    )

    embedder = Embedder(provider=OpenAIProvider())
    queue: asyncio.Queue = asyncio.Queue()

    coordinator = Coordinator(
        queue=queue,
        transform=registry.transform_fn,
        embedder=embedder,
        sync_engine=sync_engine,
        schema_class=pipeline.CHUNK_SCHEMA,
        primary_key=pipeline.PRIMARY_KEY,
    )

    monitor = Monitor(queue, manifest, transform_tracker=registry.tracker)
    monitor.add_watcher(LocalFileWatcher(pipeline.WATCHED_DIR))
    startup_done = asyncio.Event()

    monitor_task = asyncio.create_task(monitor.run(force_reindex, startup_done))
    coordinator_task = asyncio.create_task(coordinator.run())

    await startup_done.wait()
    log.info("Startup complete — watching %s for changes", pipeline.WATCHED_DIR)

    await asyncio.gather(monitor_task, coordinator_task)
