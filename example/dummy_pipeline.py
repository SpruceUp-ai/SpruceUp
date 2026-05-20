from dataclasses import dataclass, field
import re
import pathlib


@dataclass
class OutputChunk:
    lecture_title: str
    content: str
    embedding: list[float]          # 1536-dim float32
    id: int | None = field(default=None)
    # id=None → let the DB assign a serial value
    # id=<int> → hash-based ID for upsert-safe ingestion (see generate_chunk_id below)


def parse_slides(text: str) -> tuple[str, list[tuple[int, str]]]:
    """
    Parse a cleaned .txt slide file into (lecture_title, [(slide_num, text), ...]).

    Returns an empty slides list if the file doesn't follow the [Slide N] format,
    in which case the caller should fall back to paragraph-level chunking.
    """
    blocks = text.strip().split("\n\n")
    lecture_title = ""
    slides = []
    for block in blocks:
        block = block.strip()
        if block.startswith("# "):
            lecture_title = block[2:].strip()
        else:
            m = re.match(r"^\[Slide (\d+)\]\s+(.*)", block, re.DOTALL)
            if m:
                slides.append((int(m.group(1)), m.group(2).strip()))
    return lecture_title, slides


def extract_course_from_filename(filename: str) -> str:
    """Extract course code from filename, e.g. 'PHIL_1000_-_01_-_Intro.txt' → 'PHIL 1000'."""
    m = re.match(r"^([A-Z]+_\d+)", filename)
    return m.group(1).replace("_", " ") if m else "Unknown"

def sliding_window_chunks(
    lecture_title: str,
    course: str,
    slides: list[tuple[int, str]],
    window: int = 4,
    stride: int = 2,
) -> list[tuple[str, str]]:
    """
    Return (content, enriched_content) pairs for each sliding-window chunk.

    content        — raw text stored in DB and returned to the LLM
    enriched_content — metadata-prefixed text used for embedding

    Default window=4, stride=2 means chunks overlap by 2 slides.
    """
    results = []
    for start in range(0, len(slides), stride):
        window_slides = slides[start : start + window]
        content = "\n\n".join(f"[Slide {num}] {text}" for num, text in window_slides)
        enriched = f"Course: {course}\nLecture: {lecture_title}\n{content}"
        results.append((content, enriched))
    return results


def chunk_txt_file(text: str, filename: str) -> list[tuple[str, str, str]]:
    """
    Full pipeline for a .txt slide deck file.

    Returns a list of (lecture_title, content, enriched_content) tuples ready for embedding.
    Falls back to paragraph chunking if no [Slide N] blocks are found.
    """
    lecture_title, slides = parse_slides(text)
    course = extract_course_from_filename(filename)

    if slides:
        return [
            (lecture_title, content, enriched)
            for content, enriched in sliding_window_chunks(lecture_title, course, slides)
        ]

    # Fallback: no slide markers — treat as plain text, chunk by paragraph
    lecture_title = pathlib.Path(filename).stem.replace("_", " ")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [
        (lecture_title, p, f"Course: {course}\nLecture: {lecture_title}\n{p}")
        for p in paragraphs
    ]

def chunk_qa_md(text: str) -> list[tuple[str, str, str]]:
    """
    Full pipeline for a .md Q&A document.

    Returns a list of (lecture_title, content, enriched_content) tuples ready for embedding.
    Returns an empty list if no parseable structure is found.
    """
    results = []

    # --- Primary format: ## Question blocks with **Lecture:** metadata ---
    blocks = re.split(r'\n(?=## )', text.strip())
    qa_blocks = [b.strip() for b in blocks if b.strip().startswith("## ")]

    for block in qa_blocks:
        lines = block.splitlines()
        title = re.sub(r'^##\s*\*?\*?(.+?)\*?\*?\s*$', r'\1', lines[0]).strip()

        lecture_title = None
        body_start = 1
        for i, line in enumerate(lines[1:], 1):
            m = re.match(r'^\*\*Lecture:\*\*\s+(.+)$', line.strip())
            if m:
                lecture_title = m.group(1).strip()
                body_start = i + 1
                break

        if not lecture_title:
            continue

        body = "\n".join(lines[body_start:]).strip()
        if not body:
            continue

        content = f"Q: {title}\n\n{body}"
        enriched = f"Lecture: {lecture_title}\n{content}"
        results.append((lecture_title, content, enriched))

    if results:
        return results

    # --- Fallback: supplement/reading doc with # Title and plain paragraphs ---
    lines = text.strip().splitlines()
    lecture_title = None
    body_lines = []
    for line in lines:
        m = re.match(r'^#+\s+\*?\*?(.+?)\*?\*?\s*$', line)
        if m and lecture_title is None:
            lecture_title = m.group(1).strip()
        else:
            body_lines.append(line)

    if not lecture_title:
        return []

    body_text = "\n".join(body_lines)
    paragraphs = [p.strip() for p in body_text.split("\n\n") if p.strip()]
    return [
        (lecture_title, para, f"Lecture: {lecture_title}\n{para}")
        for para in paragraphs
    ]


"""
import pathlib
from openai import AsyncOpenAI

# from this file's helpers:
# OutputChunk, chunk_txt_file, chunk_qa_md, embed_content, generate_chunk_id

async def run_ingestion(corpus_dir: pathlib.Path, openai: AsyncOpenAI) -> list[OutputChunk]:
    results: list[OutputChunk] = []

    for path in corpus_dir.rglob("*.txt"):
        for lecture_title, content, enriched in chunk_txt_file(path.read_text(), path.name):
            embedding = await embed_content(openai, enriched)
            results.append(OutputChunk(
                id=generate_chunk_id(enriched),
                lecture_title=lecture_title,
                content=content,
                embedding=embedding,
            ))

    for path in corpus_dir.rglob("*.md"):
        for lecture_title, content, enriched in chunk_qa_md(path.read_text()):
            embedding = await embed_content(openai, enriched)
            results.append(OutputChunk(
                id=generate_chunk_id(enriched),
                lecture_title=lecture_title,
                content=content,
                embedding=embedding,
            ))

    return results
"""