import asyncio
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
    WeaviateTarget,
    VoyageAIEmbedder,
    defineConfig,
    memoize,
)

dotenv.load_dotenv()

# --- credentials (hardcoded for local testing) ------------------------

_GOOGLE_DRIVE_TOKEN = ""
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
_GDRIVE_FOLDER_ID = ""
_PG_CONNSTR = os.getenv("PG_CONNSTR") or ""


def get_google_drive_token() -> str:
    return _GOOGLE_DRIVE_TOKEN


# --- schema -----------------------------------------------------------


@dataclass
class LectureChunk:
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


# @memoize(return_type=str)
# async def prepare_chunk(chunk_text: str) -> str:
#     # await asyncio.sleep(0.05)  # simulate async preprocessing
#     return chunk_text


_openai_client: openai.AsyncOpenAI | None = None


def _get_openai_client() -> openai.AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


# @memoize(return_type=str)
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
    chunk_strs = split_chunks(file_props.raw_content, file_props.display_name, ext)
    # chunk_strs = [await prepare_chunk(s) for s in raw_chunks]

    title, _, _ = file_props.display_name.rpartition(".")
    embeddings = await embed(chunk_strs)
    return [
        LectureChunk(
            chunk_text=text,
            chunk_embedding=embedding,
            lecture_title=title or file_props.display_name,
        )
        for text, embedding in zip(chunk_strs, embeddings)
    ]


# --- config -----------------------------------------------------------

config = defineConfig(
    sources=[
        LocalFilesSource(watched_dir="example/data_corpus"),
        # GoogleDriveSource(
        #     watched_dir=_GDRIVE_FOLDER_ID,
        #     on_token_expired=get_google_drive_token,
        # ),
    ],
    target=PgVectorTarget(
        connstr=_PG_CONNSTR,
        table="data_chunks",
        schema=LectureChunk,
        vector_column="chunk_embedding",
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
#         primary_key="chunk_id",
#     ),
#     embedder=OpenAIEmbedder(
#         api_key=os.getenv("OPENAI_API_KEY"),
#         model="text-embedding-3-small",
#     ),
#     transform=build_lecture_chunks,
# )

# config = defineConfig(
#     sources=[
#         LocalFilesSource(watched_dir="example/data_corpus"),
#     ],
#     target=WeaviateTarget(
#         url="http://localhost:8080",
#         collection_name="dataChunks",
#         schema=LectureChunk,
#         primary_key="chunk_id",
#     ),
#     embedder=OpenAIEmbedder(
#         api_key=_OPENAI_API_KEY,
#         model="text-embedding-3-small",
#     ),
#     transform=build_lecture_chunks,
# )
