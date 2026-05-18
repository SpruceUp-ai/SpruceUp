"""
Example SpruceUp pipeline for a lecture-notes corpus.

Define your two transform functions using the provided decorators, then set
the configuration constants below.

  @file_transform   — parses a file's content into embeddable chunk strings
  @chunk_transform  — builds schema objects from those chunk strings
                      (chunk_embedding is left empty; the framework fills it in)
"""

import hashlib
import pathlib
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dummy_pipeline.dummy_pipeline import chunk_qa_md, chunk_txt_file
from registry import chunk_transform, file_transform


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
def chunk_content(content: str, filename: str) -> list[str]:
    """Parse and chunk a file into embeddable text strings."""
    ext = pathlib.Path(filename).suffix.lower()
    if ext == ".txt":
        triples = chunk_txt_file(content, pathlib.Path(filename).name)
        return [enriched for _, _, enriched in triples]
    if ext == ".md":
        triples = chunk_qa_md(content)
        return [enriched for _, _, enriched in triples]
    # Fallback: split by double-newline
    return [p.strip() for p in content.split("\n\n") if p.strip()]


@chunk_transform
def build_chunks(chunk_strs: list[str]) -> list[LectureChunk]:
    """Build LectureChunk objects from chunk strings (chunk_embedding filled by framework)."""
    chunks = []
    for text in chunk_strs:
        chunk_id = hashlib.blake2b(text.encode(), digest_size=16).hexdigest()
        chunks.append(LectureChunk(
            id=chunk_id,
            chunk_text=text,
            chunk_embedding=[],
            lecture_title="",
        ))
    return chunks


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNK_SCHEMA = LectureChunk
TARGET_DB = "spruce_lecture_rag"
TARGET_TABLE = "data_chunks"
PRIMARY_KEY = "id"
WATCHED_DIR = "dummy_pipeline/data_corpus"
