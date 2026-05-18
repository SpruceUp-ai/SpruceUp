"""
Demo / smoke-test for SyncEngine.

Patches psycopg.connect so no real Postgres instance is required.
The SQLite manifest runs for real against a temp file so you can
observe its state change after each operation.

Three scenarios:
  1. Initial reconcile  – all chunks are new
  2. Partial update     – one chunk changed, one orphaned, one added
  3. delete_file        – remove a file and all its vectors
"""

import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))

# Patch psycopg.connect BEFORE importing sync_engine so every call
# to psycopg.connect (including those inside sync_engine.py) hits our mock.
import psycopg  # noqa: E402


class _MockPgConn:
    """Prints the SQL that would be sent to Postgres instead of executing it."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, params=None):
        print(f"    SQL> {sql.strip()}")
        if params:
            print(f"         params: {_fmt(list(params))}")

    def executemany(self, sql, rows):
        print(f"    SQL> {sql.strip()}")
        for row in rows:
            print(f"         row:    {_fmt(row)}")


def _fmt(values: list) -> list:
    """Truncate long lists (e.g. embeddings) so output stays readable."""
    return [v[:3] + ["..."] if isinstance(v, list) and len(v) > 3 else v for v in values]


psycopg.connect = lambda connstr, **kwargs: _MockPgConn()

from Capstone.SpruceUp.sync_engine.sync_engine import (  # noqa: E402
    ChunkWrapper,
    File,
    SyncEngine,
    UserDefinedChunkSchema,
    hash_chunk_id,
    hash_file_path,
    hash_object,
)


# ---------------------------------------------------------------------------
# User-defined schema – what a real caller of this library would write
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk(UserDefinedChunkSchema):
    """Extends the base schema with a source_page field."""
    source_page: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(file_path: str, chunk_id: str, text: str, page: int, ordinal: int) -> ChunkWrapper:
    # 3-float embeddings (would normally be 1536-dimensional for text-embedding-3-small)
    embedding = [round(ordinal * 0.1 + i * 0.01, 3) for i in range(3)]
    user_chunk = DocumentChunk(
        id=chunk_id,
        chunk_text=text,
        chunk_embedding=embedding,
        source_page=page,
    )
    return ChunkWrapper(
        user_chunk=user_chunk,
        user_chunk_object_hash=hash_object(user_chunk),
        ordinal=ordinal,
        chunk_id=hash_chunk_id(file_path, ordinal),
    )


def make_file(file_path: str, file_type: str, mtime: float, chunks: list[ChunkWrapper]) -> File:
    fake_hash = hash_file_path(file_path)  # stand-in for real content/transform hashes
    return File(
        file_id=hash_file_path(file_path),
        file_path=file_path,
        mtime=mtime,
        content_hash=fake_hash,
        transform_hash=fake_hash,
        file_type=file_type,
        data_source_id=1,
        chunks=chunks,
    )


def print_manifest(manifest_path: str) -> None:
    with sqlite3.connect(manifest_path) as conn:
        file_rows = conn.execute(
            "SELECT hex(id), mtime, file_type FROM files ORDER BY hex(id)"
        ).fetchall()
        chunk_rows = conn.execute(
            "SELECT hex(id), hex(file_id), hex(user_chunk_object_hash) FROM chunks ORDER BY hex(id)"
        ).fetchall()

    print("  files table:")
    if not file_rows:
        print("    (empty)")
    for id_hex, mtime, file_type in file_rows:
        print(f"    file={id_hex[:16]}...  mtime={mtime}  type={file_type!r}")

    print("  chunks table:")
    if not chunk_rows:
        print("    (empty)")
    for id_hex, file_hex, hash_hex in chunk_rows:
        print(f"    chunk={id_hex[:16]}...  file={file_hex[:10]}...  obj_hash={hash_hex[:10]}...")


def section(title: str) -> None:
    bar = "=" * 62
    print(f"\n{bar}\n  {title}\n{bar}")


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

LEGAL_PATH  = "corpus/legal.pdf"
README_PATH = "corpus/README.md"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_demo() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        manifest_path = f.name

    try:
        engine = SyncEngine(manifest_path=manifest_path, pg_connstr="dbname=rag")
        engine.define_target_table(
            db_name="rag",
            table_name="vectors",
            schema_from_class=DocumentChunk,
            primary_key="id",
        )

        # -------------------------------------------------------------------
        section("SCENARIO 1: Initial reconcile — all chunks are new")
        # -------------------------------------------------------------------
        # First run ever. Manifest is empty, so every chunk is a net-new upsert.
        # legal.pdf  → 3 chunks
        # README.md  → 2 chunks
        # Expected:    5 upserts, 0 deletes; both file rows written to manifest

        legal_v1 = make_file(LEGAL_PATH, "pdf", mtime=1_000_000.0, chunks=[
            make_chunk(LEGAL_PATH, "legal_c1", "Jurisdiction clause...", page=1, ordinal=1),
            make_chunk(LEGAL_PATH, "legal_c2", "Indemnity clause...",    page=2, ordinal=2),
            make_chunk(LEGAL_PATH, "legal_c3", "Arbitration clause...",  page=3, ordinal=3),
        ])
        readme_v1 = make_file(README_PATH, "md", mtime=1_000_001.0, chunks=[
            make_chunk(README_PATH, "readme_c1", "Installation guide...", page=1, ordinal=1),
            make_chunk(README_PATH, "readme_c2", "Usage examples...",     page=1, ordinal=2),
        ])

        print("\n  >> Postgres calls:")
        engine.reconcile([legal_v1, readme_v1])

        print("\n  >> Manifest state:")
        print_manifest(manifest_path)

        # -------------------------------------------------------------------
        section("SCENARIO 2: legal.pdf updated — mixed case")
        # -------------------------------------------------------------------
        # legal_c1  same text  → hash unchanged → skip (no upsert)
        # legal_c2  new text   → hash changes   → upsert
        # legal_c3  absent     → orphan          → delete
        # legal_c4  new chunk  → net new         → upsert
        # README.md not in this call → untouched
        # Expected: 2 upserts (legal_c2, legal_c4), 1 delete (legal_c3);
        #           legal.pdf file row updated with new mtime

        legal_v2 = make_file(LEGAL_PATH, "pdf", mtime=1_000_500.0, chunks=[
            make_chunk(LEGAL_PATH, "legal_c1", "Jurisdiction clause...",         page=1, ordinal=1),  # unchanged
            make_chunk(LEGAL_PATH, "legal_c2", "Indemnity clause (amended)...",  page=2, ordinal=2),  # changed
            make_chunk(LEGAL_PATH, "legal_c4", "Confidentiality clause...",      page=4, ordinal=4),  # new
            # legal_c3 intentionally absent → will be treated as orphan
        ])

        print("\n  >> Postgres calls:")
        engine.reconcile([legal_v2])

        print("\n  >> Manifest state:")
        print_manifest(manifest_path)

        # -------------------------------------------------------------------
        section("SCENARIO 3: delete_file — README.md removed from corpus")
        # -------------------------------------------------------------------
        # Expected: delete readme_c1, readme_c2 from Postgres;
        #           README.md file row and chunk rows removed from manifest

        print("\n  >> Postgres calls:")
        engine.delete_file(hash_file_path(README_PATH))

        print("\n  >> Manifest state:")
        print_manifest(manifest_path)

        section("Demo complete")

    finally:
        os.unlink(manifest_path)


if __name__ == "__main__":
    run_demo()
