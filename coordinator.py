from dataclasses import dataclass
import os

class FileJob:
    path: str
    source: str = "local"

class SpruceFile:
    def __init__(
        self,
        path: str,
        content: str | bytes,
        mtime: float,
        content_hash: bytes,
        transform_hash: bytes,
        file_type: str,
        data_source_id: int,
        parsed_content: str | None,
        chunk_strings: str | None,
        chunks: list[str] | None,
        # chunks: list[ChunkWrapper] | None

    ):
        self._path = path
        self._content = content
        self._mtime = mtime
        self._content_hash = content_hash
        self._transform_hash = transform_hash
        self._file_type = file_type
        self._data_source_id = data_source_id
        self._parsed_content = parsed_content
        self._chunk_strings = chunk_strings
        self._chunks = chunks

    def __repr__(self):
        return f"FileObject(path={self._path!r}, content={self._content!r})"

    @property
    def path(self) -> str:
        return self._path

    @property
    def content(self) -> str | bytes:
        return self._content

    @property
    def mtime(self) -> float:
        return self._mtime

    @property
    def content_hash(self) -> bytes:
        return self._content_hash

    @property
    def transform_hash(self) -> bytes:
        return self._transform_hash

    @property
    def file_type(self) -> str:
        return self._file_type

    @property
    def data_source_id(self) -> int:
        return self._data_source_id

    @property
    def parsed_content(self) -> str | None:
        return self._parsed_content

    @property
    def chunk_strings(self) -> str | None:
        return self._chunk_strings

    @property
    def chunks(self) -> list[str] | None:
        return self._chunks

class FileFetcher:
    async def fetch(self) -> SpruceFile:
        raise NotImplementedError

class LocalFileFetcher(FileFetcher):
    def __init__(self, path: str):
        self._path = path

    async def fetch(self) -> SpruceFile:
        with open(self._path) as f:
            content = f.read()
            mtime = os.path.getmtime(self._path)
            # content_hash =
            # transform_hash =
            # file_type =
            # data_source_id =
            # parsed_content =
            # chunk_strings =
            # chunks =

        return SpruceFile(
            path=self._path,
            content=content,
            mtime=mtime,
            # content_hash=None,
            # transform_hash=None,
            # file_type=None,
            # data_source_id=None,
            # parsed_content=None,
            # chunk_strings=None,
            # chunks=None,
        )

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

        # Once `chunks` is same length as `chunk_string`, meaning all chunks have been
        # generated for the file, sync the manifest and embeddings with the sync engine.
        await self._sync_engine.reconcile(file_object, chunks, embeddings)

    async def run(self):
        while True:
            await self.process_file()
