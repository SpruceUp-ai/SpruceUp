import asyncio
from .tasks import SyncTask


class DataRetriever:
    def __init__(self, task: SyncTask):
        self._task = task

    async def retrieve(self) -> "FileObject":
        if self._task.source_type == "local":
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, self._read_local)
            return FileObject(content)
        raise NotImplementedError(f"unsupported source_type: {self._task.source_type!r}")

    def _read_local(self) -> bytes:
        with open(self._task.identifier, "rb") as f:
            return f.read()


class FileObject:
    def __init__(self, content: bytes):
        self._content = content

    def read(self) -> bytes:
        return self._content
