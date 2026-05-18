import hashlib
import os
import pathlib

from models import ChunkWrapper, SpruceFile
from monitoring.tasks import SyncTask
from sync_engine import SyncEngine, hash_chunk_id, hash_file_path, hash_object


class LocalFileFetcher:
    def __init__(self, task: SyncTask, data_source_id: int, transform_hash: bytes):
        self._task = task
        self._data_source_id = data_source_id
        self._transform_hash = transform_hash

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
            transform_hash=self._transform_hash,
            file_type=file_type,
            data_source_id=self._data_source_id,
            raw_content=raw_content,
            parsed_content=None,
            chunk_strs=[],
            chunks=[],
        )


class FetcherRegistry:
    def for_task(self, task: SyncTask, data_source_id: int, transform_hash: bytes) -> LocalFileFetcher:
        match task.source_type:
            case "local":
                return LocalFileFetcher(task, data_source_id, transform_hash)
            case _:
                raise ValueError(f"Unknown source type: {task.source_type!r}")


class Coordinator:
    """Long-lived service that pulls SyncTasks from the queue and drives the pipeline."""

    def __init__(
        self,
        queue: object,
        chunk_content,
        build_chunks,
        embedder,
        sync_engine: SyncEngine,
        transform_hash: bytes,
        data_source_id: int = 1,
    ):
        self._queue = queue
        self._chunk_content = chunk_content
        self._build_chunks = build_chunks
        self._embedder = embedder
        self._sync_engine = sync_engine
        self._transform_hash = transform_hash
        self._data_source_id = data_source_id
        self._fetcher_registry = FetcherRegistry()

    async def process_task(self) -> None:
        task: SyncTask = await self._queue.get()

        if task.change_type == "delete":
            self._sync_engine.delete_file(hash_file_path(task.identifier))
            return

        if task.change_type == "move":
            self._sync_engine.move_file(task.old_identifier, task.identifier)
            return

        # upsert: full pipeline
        fetcher = self._fetcher_registry.for_task(task, self._data_source_id, self._transform_hash)
        spruce_file = await fetcher.fetch()

        chunk_strs = self._chunk_content(spruce_file.raw_content.decode(errors="replace"), task.identifier)
        schema_objs = self._build_chunks(chunk_strs)
        embeddings = await self._embedder.process_chunks(chunk_strs)

        for obj, embedding in zip(schema_objs, embeddings):
            obj.chunk_embedding = embedding

        chunks = [
            ChunkWrapper(
                user_chunk=obj,
                user_chunk_object_hash=hash_object(obj),
                ordinal=i,
                chunk_id=hash_chunk_id(task.identifier, i),
            )
            for i, obj in enumerate(schema_objs)
        ]

        spruce_file.chunk_strs = chunk_strs
        spruce_file.chunks = chunks

        self._sync_engine.reconcile([spruce_file])

    async def run(self) -> None:
        while True:
            await self.process_task()
