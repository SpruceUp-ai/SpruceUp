import asyncio
import hashlib
import os
from dataclasses import dataclass

import dotenv
import openai

from example.dummy_pipeline import chunk_qa_md, chunk_txt_file
from spruceup import (
    CohereEmbedder,
    FileProps,
    GeminiEmbedder,
    GoogleDriveSource,
    LocalFilesSource,
    OpenAIEmbedder,
    PgVectorTarget,
    PineconeTarget,
    VoyageAIEmbedder,
    WeaviateTarget,
    defineConfig,
    memoize,
)

dotenv.load_dotenv()

# --- credentials (hardcoded for local testing) ------------------------

_GOOGLE_DRIVE_TOKEN = ""
_OPENAI_API_KEY = ""
_GDRIVE_FOLDER_ID = ""
_PG_CONNSTR = ""


def get_google_drive_token() -> str:
    return _GOOGLE_DRIVE_TOKEN


# --- schema -----------------------------------------------------------


@dataclass
class LectureChunk:
    # id: str  # reserved by Weaviate; use chunk_id instead when targeting Weaviate
    chunk_id: str
    chunk_text: str
    chunk_embedding: list[float]
    # chunk_summary: str
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
async def prepare_chunk(chunk_text: str) -> str:
    # await asyncio.sleep(0.05)  # simulate async preprocessing
    return chunk_text


_openai_client: openai.AsyncOpenAI | None = None


def _get_openai_client() -> openai.AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


# @memoize(returns=str)
# async def summarize_chunk(chunk_text: str) -> str:
#     response = await _get_openai_client().chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": f"Summarize in less than 10 words:\n\n{chunk_text}"}],
#         max_tokens=20,
#     )
#     return response.choices[0].message.content.strip()


# --- transform --------------------------------------------------------


async def build_lecture_chunks(*, file_props: FileProps, embed) -> list[LectureChunk]:
    ext = "." + file_props.file_type if file_props.file_type else ""
    raw_chunks = split_chunks(file_props.raw_content, file_props.display_name, ext)
    chunk_strs = [await prepare_chunk(s) for s in raw_chunks]

    title, _, _ = file_props.display_name.rpartition(".")
    embeddings = await embed(chunk_strs)
    return [
        LectureChunk(
            # id=hashlib.blake2b(text.encode(), digest_size=16).hexdigest(),
            chunk_id=hashlib.blake2b(text.encode(), digest_size=16).hexdigest(),
            chunk_text=text,
            chunk_embedding=embedding,
            lecture_title=title or file_props.display_name,
        )
        for text, embedding in zip(chunk_strs, embeddings)
    ]


# --- config -----------------------------------------------------------

# config = defineConfig(
#     sources=[
#         LocalFilesSource(watched_dir="example/data_corpus"),
#         # GoogleDriveSource(
#         #     folder_id="1QY9VJYPpKtIQsCBvl-SsxZf6CHJ601t5",
#         #     on_token_expired=lambda: os.getenv("GOOGLE_DRIVE_TOKEN"),
#         # ),
#     ],
#     target=PgVectorTarget(
#         connstr=os.getenv("PG_CONNSTR"),
#         table="data_chunks",             # original table
#         # table="data_chunks_voyageai",    # table for default 1024 dim vectors
#         # table="data_chunks_voyageai512", # table for 512 dim vectors
#         # table="data_chunks_cohere",      # table for cohere
#         # table="data_chunks_gemini",
#         schema=LectureChunk,
#         primary_key="id",
#     ),
#     embedder=OpenAIEmbedder(
#         api_key=os.getenv("OPENAI_API_KEY"),
#         model="text-embedding-3-small",
#     ),
#     # embedder=VoyageAIEmbedder(
#     #     api_key=os.getenv("VOYAGE_API_KEY"),
#     #     model="voyage-4-lite",
#     #     # embedding_dimensions=512
#     # ),
#     # embedder=CohereEmbedder(
#     #     api_key=os.getenv("COHERE_API_KEY"),
#     #     model="embed-v4.0"
#     # ),
#     transform=build_lecture_chunks,
#     # embedder=GeminiEmbedder(
#     #     api_key=os.getenv("GEMINI_API_KEY"),
#     #     model="gemini-embedding-001"
#     # )
# )

config = defineConfig(
    sources=[
        LocalFilesSource(watched_dir="example/data_corpus"),
        GoogleDriveSource(
            watched_dir=_GDRIVE_FOLDER_ID,
            on_token_expired=get_google_drive_token,
        ),
    ],
    target=PgVectorTarget(
        connstr=_PG_CONNSTR,
        table="data_chunks",
        schema=LectureChunk,
        primary_key="chunk_id",
    ),
    embedder=OpenAIEmbedder(
        api_key=_OPENAI_API_KEY,
        model="text-embedding-3-small",
    ),
    transform=build_lecture_chunks,
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
