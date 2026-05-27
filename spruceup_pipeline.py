import hashlib
import os
import pathlib
from dataclasses import dataclass

from example.dummy_pipeline import chunk_qa_md, chunk_txt_file
from spruceup import LocalFilesSource, OpenAIEmbedder, PgVectorTarget, VoyageAIEmbedder, CohereEmbedder, GeminiEmbedder, defineConfig, memoize
from spruceup import PineconeTarget

import dotenv

dotenv.load_dotenv()

# --- schema -----------------------------------------------------------

@dataclass
class LectureChunk:
    id: str
    chunk_text: str
    chunk_embedding: list[float]
    lecture_title: str


# --- helpers ----------------------------------------------------------

def split_chunks(raw_content: str, file_name: str, ext: str) -> list[str]:
    if ext == ".txt":
        triples = chunk_txt_file(raw_content, file_name)
        return [enriched for _, _, enriched in triples]
    if ext == ".md":
        triples = chunk_qa_md(raw_content)
        return [enriched for _, _, enriched in triples]
    return [p.strip() for p in raw_content.split("\n\n") if p.strip()]


# --- memoized helpers -------------------------------------------------

@memoize(returns=str)
def prepare_chunk(chunk_text: str) -> str:
    return chunk_text


# --- transform --------------------------------------------------------

async def build_lecture_chunks(*, file_props: dict, embed) -> list[LectureChunk]:
    file_path = pathlib.Path(file_props["file_path"])
    raw_chunks = split_chunks(
        file_props["raw_content"], file_path.name, file_path.suffix.lower()
    )
    chunk_strs = [prepare_chunk(s) for s in raw_chunks]
    # chunk_strs = raw_chunks

    embeddings = await embed(chunk_strs)
    return [
        LectureChunk(
            id=hashlib.blake2b(text.encode(), digest_size=16).hexdigest(),
            chunk_text=text,
            chunk_embedding=embedding,
            lecture_title=file_path.stem,
        )
        for text, embedding in zip(chunk_strs, embeddings)
    ]


# --- config -----------------------------------------------------------

config = defineConfig(
    sources=[
        LocalFilesSource(watched_dir="example/data_corpus"),
        # LocalFilesSource(watched_dir="example/second_local_source"),
    ],
    target=PgVectorTarget(
        connstr=os.getenv("PG_CONNSTR"),
        # table="data_chunks",             # original table
        # table="data_chunks_voyageai",    # table for default 1024 dim vectors
        # table="data_chunks_voyageai512", # table for 512 dim vectors
        # table="data_chunks_cohere",      # table for cohere
        table="data_chunks_gemini",
        schema=LectureChunk,
        primary_key="id",
    ),
    # embedder=OpenAIEmbedder(
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     model="text-embedding-3-small",
    # ),
    # embedder=VoyageAIEmbedder(
    #     api_key=os.getenv("VOYAGE_API_KEY"),
    #     model="voyage-4-lite",
    #     # embedding_dimensions=512
    # ),
    # embedder=CohereEmbedder(
    #     api_key=os.getenv("COHERE_API_KEY"),
    #     model="embed-v4.0"
    # ),
    transform=build_lecture_chunks,
    embedder=GeminiEmbedder(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-embedding-001"
    )
)

# config = defineConfig(
#     sources=[
#         LocalFilesSource(watched_dir="example/data_corpus"),
#         # LocalFilesSource(watched_dir="example/second_local_source"),
#     ],
#     target=PineconeTarget(
#         api_key=os.getenv("PINECONE_API_KEY"),
#         index_name="data-chunks",
#         schema=LectureChunk,
#         primary_key="id",
#     ),
#     embedder=OpenAIEmbedder(
#         api_key=os.getenv("OPENAI_API_KEY"),
#         model="text-embedding-3-small",
#     ),
#     transform=build_lecture_chunks,
# )
