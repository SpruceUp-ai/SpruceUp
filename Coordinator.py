import asyncio
from Embedding import Embedder

# A queue for managing file paths to be processed.
# This is likely unnecessary until queue's logic becomes more complex than
# simply adding/removing files from the queue.
class JobQueue:
    def __init__(self):
        self._queue = asyncio.Queue()

    def add_file(self, file: str):
        """Enqueue a file path for a file to eventually be processed."""
        pass

    def get_file(self):
        """Dequeue a file path for processing."""
        pass

class FileObject:
    def __init__(self, path: str, content: str):
        self._path = path
        self._content = content

    def __repr__(self):
        return f"FileObject(path={self._path!r}, content={self._content!r})"

    @property
    def path(self) -> str:
        return self._path

    @property
    def content(self) -> str:
        return self._content

class FileFetcher:
    async def fetch_file(self, path: str, source = None) -> FileObject:
        # `source` will be necessary to determine _where_ to fetch the file from
        pass

class Coordinator:
    """
    Long-lived service object that coordinates fetching and processing of a given file.
    """
    def __init__(self, queue, parser, chunker, embedder, manifest, sync_engine):
        self._queue = queue
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._manifest = manifest
        self._sync_engine = sync_engine

    def dequeue_next_file(self):
        return self._queue.get()

    async def process_file(self):
        # dequeue the next `file` from the queue
        file_job = await self.dequeue_next_file()

        # Spawn FileFetcher to fetch `file` -> `FileObject`
        file_object = await FileFetcher().fetch_file(path=file_job.path)
        # file = await FileFetcher().fetch_file(path=file_job.path, source=file_job.source)

        # update `_manifest` with the `file` being processed
        await self._manifest.update_manifest(file_object)

        # parse file, receive parsed content
        parsed_content = await self._parser.parse_file_content(file_object.content)

        # chunk parsed content
        chunks = await self._chunker.chunk_file(parsed_content)

        # embed chunks
        embeddings = await self._embedder.process_chunks(chunks)

        # sync manifest embeddings
        await self._sync_engine.sync_manifest_and_embeddings(file_object, chunks, embeddings)

    async def run(self):
        while True:
            await self.process_file()
