import hashlib
import os
import pathlib
from dataclasses import dataclass

from example.dummy_pipeline import chunk_qa_md, chunk_txt_file
from spruceup import LocalFilesSource, OpenAIEmbedder, PgVectorTarget, defineConfig

import dotenv

dotenv.load_dotenv()

# --- schema -----------------------------------------------------------

@dataclass
class LectureChunk:
    id: str
    chunk_text: str
    chunk_embedding: list[float]
    lecture_title: str


# --- memoized helpers -------------------------------------------------

# placeholder for memoized subfunctions


# --- transform --------------------------------------------------------

async def build_lecture_chunks(*, file_props: dict, embed) -> list[LectureChunk]:
    content = file_props["raw_content"]
    ext = pathlib.Path(file_props["file_path"]).suffix.lower()

    if ext == ".txt":
        triples = chunk_txt_file(content, pathlib.Path(file_props["file_path"]).name)
        chunk_strs = [enriched for _, _, enriched in triples]
    elif ext == ".md":
        triples = chunk_qa_md(content)
        chunk_strs = [enriched for _, _, enriched in triples]
    else:
        chunk_strs = [p.strip() for p in content.split("\n\n") if p.strip()]

    embeddings = await embed(chunk_strs)
    return [
        LectureChunk(
            id=hashlib.blake2b(text.encode(), digest_size=16).hexdigest(),
            chunk_text=text,
            chunk_embedding=embedding,
            lecture_title=pathlib.Path(file_props["file_path"]).stem,
        )
        for text, embedding in zip(chunk_strs, embeddings)
    ]


# --- config -----------------------------------------------------------

config = defineConfig(
    sources=[
        LocalFilesSource(watched_dir="example/data_corpus"),
    ],
    target=PgVectorTarget(
        connstr=os.getenv("PG_CONNSTR"),
        table="data_chunks",
        schema=LectureChunk,
        primary_key="id",
    ),
    embedder=OpenAIEmbedder(
        model="text-embedding-3-small",
    ),
    transform=build_lecture_chunks,
)
