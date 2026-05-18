import hashlib
import inspect
import sqlite3
from typing import Callable

DIGEST_SIZE = 16


class TransformTracker:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._transforms: list[Callable] = []

    def __call__(self, func: Callable) -> Callable:
        self._transforms.append(func)
        return func

    def register(self, func: Callable) -> Callable:
        """Register a transform function (non-decorator form of __call__)."""
        return self(func)

    def _hash_source(self, func: Callable) -> bytes:
        return hashlib.blake2b(inspect.getsource(func).encode(), digest_size=DIGEST_SIZE).digest()

    def current_hash(self) -> bytes:
        """Return a combined BLAKE2B hash of all registered transform functions."""
        h = hashlib.blake2b(digest_size=DIGEST_SIZE)
        for func in self._transforms:
            h.update(self._hash_source(func))
        return h.digest()

    def configure(self, db_path: str) -> None:
        self._db_path = db_path

    def any_changed(self) -> bool:
        con = sqlite3.connect(self._db_path)
        for func in self._transforms:
            row = con.execute(
                "SELECT source_hash FROM transform_hashes WHERE func_name = ?",
                (func.__qualname__,),
            ).fetchone()
            if row is None or row[0] != self._hash_source(func):
                con.close()
                return True
        con.close()
        return False

    def record_all(self) -> None:
        con = sqlite3.connect(self._db_path)
        for func in self._transforms:
            con.execute(
                "INSERT INTO transform_hashes (func_name, source_hash) VALUES (?,?) "
                "ON CONFLICT(func_name) DO UPDATE SET source_hash=excluded.source_hash;",
                (func.__qualname__, self._hash_source(func)),
            )
        con.commit()
        con.close()
