import sqlite3
from typing import Callable

from hashing import hash_transform


class TransformTracker:
    def __init__(self, manifest_path: str):
        self._manifest_path = manifest_path
        self._transforms: list[Callable] = []

    def register(self, func: Callable) -> Callable:
        self._transforms.append(func)
        return func

    def configure(self, manifest_path: str) -> None:
        self._manifest_path = manifest_path

    def any_changed(self) -> bool:
        con = sqlite3.connect(self._manifest_path)
        for func in self._transforms:
            row = con.execute(
                "SELECT 1 FROM transform_hashes WHERE transform_hash = ?",
                (hash_transform(func),),
            ).fetchone()
            if row is None:
                con.close()
                return True
        con.close()
        return False

    def update_transform_hashes(self) -> None:
        current_hashes = [hash_transform(func) for func in self._transforms]
        placeholders = ",".join("?" * len(current_hashes))
        con = sqlite3.connect(self._manifest_path)
        con.execute(
            f"DELETE FROM transform_hashes WHERE transform_hash NOT IN ({placeholders})",
            current_hashes,
        )
        for h in current_hashes:
            con.execute(
                "INSERT OR IGNORE INTO transform_hashes (transform_hash) VALUES (?)",
                (h,),
            )
        con.commit()
        con.close()
