import asyncio
import importlib

import registry
from coordinator import Coordinator
from db import init_db
from embedding import Embedder, OpenAIProvider
from monitoring.monitor import LocalFileWatcher, Monitor
from sync_engine import SyncEngine

DB_PATH = "sync.db"
PG_CONNSTR = "postgresql://localhost:5432/spruceup"  # hardcoded for MVP

# Importing the pipeline triggers the @file_transform / @chunk_transform decorators,
# which register both functions with registry.tracker.
pipeline = importlib.import_module("spruceup_pipeline")

# Now that the functions are registered, point the tracker at the real db.
registry.tracker.configure(DB_PATH)


async def main() -> None:
    init_db(DB_PATH)
    force_reindex = registry.tracker.any_changed()

    sync_engine = SyncEngine(manifest_path=DB_PATH, pg_connstr=PG_CONNSTR)
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
        chunk_content=registry.file_transform_fn,
        build_chunks=registry.chunk_transform_fn,
        embedder=embedder,
        sync_engine=sync_engine,
        transform_hash=registry.tracker.current_hash(),
    )

    monitor = Monitor(queue, DB_PATH)
    monitor.add_watcher(LocalFileWatcher(pipeline.WATCHED_DIR))
    startup_done = asyncio.Event()

    monitor_task = asyncio.create_task(monitor.run(force_reindex, startup_done))
    coordinator_task = asyncio.create_task(coordinator.run())

    await startup_done.wait()
    if force_reindex:
        registry.tracker.record_all()

    await asyncio.gather(monitor_task, coordinator_task)


if __name__ == "__main__":
    asyncio.run(main())
