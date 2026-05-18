import asyncio
from coordinator import Coordinator, FetcherRegistry, FileJob
from embedding import Embedder

class StubParser:
    async def parse_file_content(self, content):
        return content  # pretend parsing — return content unchanged

class StubChunker:
    async def chunk_file(self, content):
        return [content]  # pretend chunking — single chunk

class StubManifest:
    async def update_manifest(self, file_object):
        print(f"[manifest] updated for {file_object.path}")

class StubSyncEngine:
    async def sync_manifest_and_embeddings(self, file_object, chunks, embeddings):
        print(f"[sync] {file_object.path}: {len(chunks)} chunks, {len(embeddings)} vectors")

async def main():
    # queue = asyncio.Queue()
    # fetcher_registry = FetcherRegistry()
    # coordinator = Coordinator(
    #   queue=queue,
    #  fetcher_registry=fetcher_registry,
    #  parser=parser,
    #  chunker=chunker,
    #  embedder=embedder,
    #  manifest_manifest,
    #  sync_engine=sync_engine
    # )

    # await coordinator.run()

    pass


if __name__ == "__main__":
    asyncio.run(main())
