import asyncio
import importlib
import logging

import spruceup.registry as registry
from spruceup.coordinator import Coordinator
from spruceup.db import init_db
from spruceup.embedding import Embedder, OpenAIProvider
from spruceup.monitoring.monitor import LocalFileWatcher, Monitor
from spruceup.sync_engine import SyncEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MANIFEST_PATH = "spruceup_manifest.db"
PG_CONNSTR = "postgresql://localhost:5432/spruce_lecture_rag"  # hardcoded for MVP

# Importing the pipeline triggers the @file_transform / @chunk_transform decorators,
# which register both functions with registry.tracker.
pipeline = importlib.import_module("spruceup_pipeline")

# Now that the functions are registered, point the tracker at the real db.
registry.tracker.configure(MANIFEST_PATH)


async def main() -> None:
    init_db(MANIFEST_PATH)
    log.info(
        "SpruceUp starting — manifest=%s  target=%s/%s",
        MANIFEST_PATH, pipeline.TARGET_DB, pipeline.TARGET_TABLE,
    )

    force_reindex = registry.tracker.any_changed()
    if force_reindex:
        log.info("Transform functions changed — full reindex scheduled")
    else:
        log.info("Transform functions unchanged — incremental sync")

    sync_engine = SyncEngine(manifest_path=MANIFEST_PATH, pg_connstr=PG_CONNSTR)
    sync_engine.define_target_table(
        db_name=pipeline.TARGET_DB,
        table_name=pipeline.TARGET_TABLE,
        schema_from_class=pipeline.CHUNK_SCHEMA,
        primary_key=pipeline.PRIMARY_KEY,
    )

    embedder = Embedder(provider=OpenAIProvider())
    queue: asyncio.Queue = asyncio.Queue()

    coordinator = Coordinator(
        queue=queue,
        file_transform=registry.file_transform_fn,
        chunk_transform=registry.chunk_transform_fn,
        embedder=embedder,
        sync_engine=sync_engine,
    )

    monitor = Monitor(queue, MANIFEST_PATH, transform_tracker=registry.tracker)
    monitor.add_watcher(LocalFileWatcher(pipeline.WATCHED_DIR))
    startup_done = asyncio.Event()

    monitor_task = asyncio.create_task(monitor.run(force_reindex, startup_done))
    coordinator_task = asyncio.create_task(coordinator.run())

    await startup_done.wait()
    log.info("Startup complete — watching %s for changes", pipeline.WATCHED_DIR)

    await asyncio.gather(monitor_task, coordinator_task)


if __name__ == "__main__":
    asyncio.run(main())
