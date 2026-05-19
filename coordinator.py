import hashlib
import logging
import os
import pathlib

from models import ChunkWrapper, SpruceFile
from monitoring.tasks import SyncTask
from hashing import hash_chunk_id, hash_file_path, hash_object
from sync_engine import SyncEngine

log = logging.getLogger(__name__)


class LocalFileFetcher:
    def __init__(self, task: SyncTask, data_source_id: int):
        self._task = task
        self._data_source_id = data_source_id

    async def fetch(self) -> SpruceFile:
        path = self._task.identifier
        with open(path, "rb") as f:
            raw_content = f.read()
        stat = os.stat(path)
        content_hash = hashlib.blake2b(raw_content, digest_size=16).digest()
        file_type = pathlib.Path(path).suffix.lstrip(".")
        return SpruceFile(
            file_id=hash_file_path(path),
            file_path=path,
            inode=stat.st_ino,
            mtime=stat.st_mtime,
            content_hash=content_hash,
            file_type=file_type,
            data_source_id=self._data_source_id,
            raw_content=raw_content,
            chunk_strs=[],
            chunks=[],
        )


class FetcherRegistry:
    def for_task(self, task: SyncTask, data_source_id: int) -> LocalFileFetcher:
        match task.source_type:
            case "local":
                return LocalFileFetcher(task, data_source_id)
            case _:
                raise ValueError(f"Unknown source type: {task.source_type!r}")


class Coordinator:
    """Long-lived service that pulls SyncTasks from the queue and drives the pipeline."""

    def __init__(
        self,
        queue: object,
        file_transform,
        chunk_transform,
        embedder,
        sync_engine: SyncEngine,
        data_source_id: int = 1,
    ):
        self._queue = queue
        self._file_transform = file_transform
        self._chunk_transform = chunk_transform
        self._embedder = embedder
        self._sync_engine = sync_engine
        self._data_source_id = data_source_id
        self._fetcher_registry = FetcherRegistry()

    async def process_task(self) -> None:
        task: SyncTask = await self._queue.get()
        filename = pathlib.Path(task.identifier).name

        if task.change_type == "delete":
            log.info("[delete] %s", filename)
            self._sync_engine.delete_file(task.identifier)
            return

        if task.change_type == "move":
            old_name = pathlib.Path(task.old_identifier).name
            log.info("[move] %s → %s", old_name, filename)
            self._sync_engine.move_file(task.old_identifier, task.identifier)
            return

        # upsert: full pipeline for this file
        fetcher = self._fetcher_registry.for_task(task, self._data_source_id)
        spruce_file = await fetcher.fetch()

        chunk_strs = self._file_transform(file_props={
            "raw_content": spruce_file.raw_content.decode(errors="replace"),
            "file_path": spruce_file.file_path,
            "mtime": spruce_file.mtime,
            "file_type": spruce_file.file_type,
        })
        spruce_file.chunk_strs = chunk_strs
        log.info("[upsert] %s — %d chunk(s), embedding …", filename, len(chunk_strs))
        schema_objs = await self._chunk_transform(chunk_strs, embed=self._embedder.process_chunks)

        chunks = [
            ChunkWrapper(
                user_chunk=obj,
                user_chunk_object_hash=hash_object(obj),
                ordinal=i,
                chunk_id=hash_chunk_id(task.identifier, i),
            )
            for i, obj in enumerate(schema_objs)
        ]

        spruce_file.chunks = chunks

        self._sync_engine.reconcile([spruce_file])

    async def run(self) -> None:
        while True:
            await self.process_task()
