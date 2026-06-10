import asyncio

from .models import SyncTask


class DebounceQueue(asyncio.Queue[SyncTask]):
    """asyncio.Queue that evicts an already-queued task for the same file_id."""

    def __init__(self, maxsize: int = 10000) -> None:
        super().__init__(maxsize=maxsize)
        self._in_queue: dict[str, SyncTask] = {}

    def _put(self, item: SyncTask) -> None:
        old = self._in_queue.get(item.current_file_id)
        if old is not None:
            self._queue.remove(old)  # pyright: ignore[reportAttributeAccessIssue]
            self._unfinished_tasks -= 1
        self._in_queue[item.current_file_id] = item
        super()._put(item)

    def _get(self) -> SyncTask:
        item = super()._get()
        self._in_queue.pop(item.current_file_id, None)
        return item
