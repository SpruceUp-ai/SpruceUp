import asyncio

from .models import SyncTask


class DebounceQueue(asyncio.Queue):
    """asyncio.Queue that evicts same-file tasks on enqueue.

    _in_queue tracks which file_ids are currently in the queue, enabling
    O(1) duplicate detection. On eviction the old task is removed from the
    deque and _unfinished_tasks is decremented so join() stays consistent.
    Entries are cleared from _in_queue in _get when a task is consumed.
    """

    def __init__(self, maxsize: int = 10000) -> None:
        super().__init__(maxsize=maxsize)
        self._in_queue: dict[str, SyncTask] = {}

    def _put(self, item: SyncTask) -> None:
        if item.current_file_id is not None:
            old = self._in_queue.get(item.current_file_id)
            if old is not None:
                self._queue.remove(old)
                self._unfinished_tasks -= 1
            self._in_queue[item.current_file_id] = item
        super()._put(item)

    def _get(self) -> SyncTask:
        item = super()._get()
        self._in_queue.pop(item.current_file_id, None)
        return item
