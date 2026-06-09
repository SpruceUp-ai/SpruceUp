from dataclasses import dataclass

from spruceup.utils.hashing import hash_chunk_content


@dataclass
class Chunk:
    text: str
    embedding: list[float]


def test_hash_ignores_vector_column():
    a = Chunk(text="hello", embedding=[0.1, 0.2])
    b = Chunk(text="hello", embedding=[0.9, 0.8, 0.7])
    assert hash_chunk_content(a, "embedding") == hash_chunk_content(b, "embedding")


def test_hash_changes_when_content_changes():
    a = Chunk(text="hello", embedding=[0.1, 0.2])
    b = Chunk(text="world", embedding=[0.1, 0.2])
    assert hash_chunk_content(a, "embedding") != hash_chunk_content(b, "embedding")
