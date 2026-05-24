from typing import Callable

from ..utils.hashing import hash_transform


class TransformTracker:
    """Registers transform functions and exposes their current hashes.

    Database access (checking/updating hashes in the manifest) is handled by
    Manifest.transform_hashes_changed() and Manifest.update_transform_hashes().
    """

    def __init__(self):
        self._transforms: list[Callable] = []

    def register(self, func: Callable) -> Callable:
        self._transforms.append(func)
        return func

    @property
    def hashes(self) -> list[bytes]:
        return [hash_transform(func) for func in self._transforms]
