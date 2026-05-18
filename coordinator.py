from dataclasses import dataclass

class FileJob:
    path: str
    source: str = "local"

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
    async def fetch(self) -> FileObject:
        raise NotImplementedError

class LocalFileFetcher(FileFetcher):
    def __init__(self, path: str):
        self._path = path

    async def fetch(self) -> FileObject:
        with open(self._path) as f:
            content = f.read()

        return FileObject(path=self._path, content=content)

# NOTE: Included as stub for future expansion.
# class NotionFetcher(FileFetcher):  # post-MVP
#     def __init__(self, notion_client):
#         self._client = notion_client

#     async def fetch(self) -> FileObject:
#       pass

class FetcherRegistry:
    def __init__(self, notion_client=None):
        self._notion_client = notion_client

    def for_job(self, job) -> FileFetcher:
        match job.source:
            case "local":
                return LocalFileFetcher(path=job.path)
            # case "notion":
            #     return NotionFetcher(notion_client=self._notion_client)
            case _:
                raise ValueError(f"Unknown file source: {job.source}")


class Coordinator:
    """
    Long-lived service object that coordinates fetching and processing of a given file.
    """
    def __init__(self, queue, fetcher_registry, parser, chunker, embedder, manifest, sync_engine):
        self._queue = queue
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._manifest = manifest
        self._sync_engine = sync_engine
        self._fetcher_registry = fetcher_registry

    async def process_file(self):
        file_job = await self._queue.get()
        fetcher = self._fetcher_registry.for_job(file_job)
        file_object = await fetcher.fetch()

        await self._manifest.update_manifest(file_object)

        parsed_content = await self._parser.parse_file_content(file_object.content)
        chunks = await self._chunker.chunk_file(parsed_content)
        embeddings = await self._embedder.process_chunks(chunks)

        await self._sync_engine.sync_manifest_and_embeddings(file_object, chunks, embeddings)

    async def run(self):
        while True:
            await self.process_file()
