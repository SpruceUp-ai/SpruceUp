import asyncio

from sync_engine import SyncEngine

from db import init_db
from monitoring.capture import TransformTracker
from monitoring.monitor import LocalFileWatcher, Monitor

DB_PATH = "sync.db"

tracker = TransformTracker(DB_PATH)


async def main() -> None:
    init_db(DB_PATH)
    force_reindex = tracker.any_changed()
    queue = asyncio.Queue()
    sync_engine = SyncEngine()
    monitor = Monitor(queue, DB_PATH, sync_engine)
    monitor.add_watcher(LocalFileWatcher("monitoring/test_files"))
    startup_done = asyncio.Event()
    monitor_task = asyncio.create_task(monitor.run(force_reindex, startup_done))
    await startup_done.wait()
    if force_reindex:
        tracker.record_all()
    await monitor_task


if __name__ == "__main__":
    asyncio.run(main())
