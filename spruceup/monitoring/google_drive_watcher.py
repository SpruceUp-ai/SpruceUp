import asyncio
from collections.abc import Callable

from ..manifest import Manifest
from .monitor import BaseWatcher, _BufferedQueue


class GoogleDriveWatcher(BaseWatcher):
    def __init__(
        self,
        folder_id: str,
        data_source_id: int,
        source_type: str,
        on_token_expired: Callable[[], str],
        poll_interval: float = 30.0,
    ):
        self._folder_id = folder_id
        self._data_source_id = data_source_id
        self._source_type = source_type
        self._on_token_expired = on_token_expired

    @property
    def source_type(self) -> str:
        return self._source_type

    async def _catch_up(
        self,
        queue: asyncio.Queue,
        manifest: "Manifest",
        force_reindex: bool = False,
    ) -> None: ...

    async def _watch(
        self,
        queue: "_BufferedQueue",
        manifest: "Manifest",
    ) -> None: ...
