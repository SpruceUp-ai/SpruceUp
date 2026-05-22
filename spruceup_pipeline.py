"""
Example SpruceUp pipeline for a lecture-notes corpus.

Define your transform function using the provided decorator, then call
defineConfig() with your source, target, and embedding connectors.

  @transform  — converts a file into a list of your schema objects;
                call embed(chunk_strs) to get embeddings and assign them yourself
"""

import hashlib
import os
import pathlib
from dataclasses import dataclass

from example.dummy_pipeline import chunk_qa_md, chunk_txt_file
from spruceup import LocalFilesSource, OpenAIEmbedder, PgVectorTarget, defineConfig, transform


# ---------------------------------------------------------------------------
# User-defined schema
# ---------------------------------------------------------------------------

@dataclass
class LectureChunk:
    id: str
    chunk_text: str
    chunk_embedding: list[float]
    lecture_title: str


# ---------------------------------------------------------------------------
# Transform function
# ---------------------------------------------------------------------------

@transform
async def build_lecture_chunks(*, file_props: dict, embed) -> list[LectureChunk]:
    """Parse a file into chunks, embed them, and return LectureChunk objects."""
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
            lecture_title="",
        )
        for text, embedding in zip(chunk_strs, embeddings)
    ]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

config = defineConfig(
    sources=[
        LocalFilesSource(watched_dir="example/data_corpus"),
    ],
    target=PgVectorTarget(
        connstr=os.environ["PG_CONNSTR"],
        table="data_chunks",
        schema=LectureChunk,
        primary_key="id",
    ),
    embeddings=OpenAIEmbedder(
        model="text-embedding-3-small",
    ),
)
