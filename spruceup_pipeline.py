"""
Example SpruceUp pipeline for a lecture-notes corpus.

Define your two transform functions using the provided decorators, then set
the configuration constants below.

  @file_transform   — parses a file's content into embeddable chunk strings
  @chunk_transform  — builds schema objects from those chunk strings;
                      call embed(chunk_strs) to get embeddings and assign them yourself
"""

import hashlib
import pathlib
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from example.dummy_pipeline import chunk_qa_md, chunk_txt_file
from spruceup.registry import chunk_transform, file_transform


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
# Transform functions
# ---------------------------------------------------------------------------

@file_transform
def chunk_content(*, file_props: dict) -> list[str]:
    """Parse and chunk a file into embeddable text strings."""
    content = file_props["raw_content"]
    ext = pathlib.Path(file_props["file_path"]).suffix.lower()
    if ext == ".txt":
        triples = chunk_txt_file(content, pathlib.Path(file_props["file_path"]).name)
        return [enriched for _, _, enriched in triples]
    if ext == ".md":
        triples = chunk_qa_md(content)
        return [enriched for _, _, enriched in triples]
    # Fallback: split by double-newline
    return [p.strip() for p in content.split("\n\n") if p.strip()]


@chunk_transform
async def build_chunks(chunk_strs: list[str], *, embed) -> list[LectureChunk]:
    """Build LectureChunk objects from chunk strings, embedding them via the provided callable."""
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

CHUNK_SCHEMA = LectureChunk
TARGET_DB = "spruce_lecture_rag"
TARGET_TABLE = "data_chunks"
PRIMARY_KEY = "id"
WATCHED_DIR = "example/data_corpus"
