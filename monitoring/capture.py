import hashlib
import inspect
import sqlite3
from typing import Callable


class TransformTracker:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._transforms: list[Callable] = []

    def __call__(self, func: Callable) -> Callable:
        self._transforms.append(func)
        return func

    def _hash_source(self, func: Callable) -> bytes:
        return hashlib.sha256(inspect.getsource(func).encode()).digest()

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
