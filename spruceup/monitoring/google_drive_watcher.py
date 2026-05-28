import asyncio
import json
import logging
from collections.abc import Callable

from ..manifest import Manifest
from ..models import SyncTask
from .monitor import BaseWatcher

log = logging.getLogger(__name__)

_FOLDER_MIME = "application/vnd.google-apps.folder"
_STATE_PAGE_TOKEN = "page_token"
_STATE_FOLDER_IDS = "watched_folder_ids"


class GoogleDriveWatcher(BaseWatcher):
    def __init__(
        self,
        folder_id: str,
        data_source_id: int,
        source_type: str,
        on_token_expired: Callable[[], str],
        is_supported: Callable[[str], bool],
        poll_interval: float = 30.0,
    ):
        self._folder_id = folder_id
        self._data_source_id = data_source_id
        self._source_type = source_type
        self._on_token_expired = on_token_expired
        self._is_supported = is_supported
        self._poll_interval = poll_interval

    @property
    def source_type(self) -> str:
        return self._source_type

    def _build_service(self):
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials

        try:
            token = self._on_token_expired()
        except Exception as exc:
            raise RuntimeError(
                "GoogleDriveWatcher: on_token_expired() raised an error — "
                "ensure it returns a valid access token string."
            ) from exc
        if not token:
            raise RuntimeError(
                "GoogleDriveWatcher: on_token_expired() returned an empty token."
            )
        return build("drive", "v3", credentials=Credentials(token=token))

    async def _full_scan(
        self, service, queue: asyncio.Queue, manifest: "Manifest"
    ) -> int:
        """BFS the folder tree, enqueue upserts, anchor the Changes cursor. Returns upsert count."""
        all_folder_ids: set[str] = {self._folder_id}
        folders_to_scan = [self._folder_id]
        n_upserts = 0

        while folders_to_scan:
            folder_id = folders_to_scan.pop()
            page_token = None

            while True:
                list_kwargs = dict(
                    q=f"'{folder_id}' in parents and trashed = false",
                    fields="nextPageToken, files(id, mimeType)",
                    pageSize=100,
                )
                if page_token:
                    list_kwargs["pageToken"] = page_token

                response = await asyncio.to_thread(
                    service.files().list(**list_kwargs).execute
                )
                for f in response.get("files", []):
                    if f["mimeType"] == _FOLDER_MIME:
                        all_folder_ids.add(f["id"])
                        folders_to_scan.append(f["id"])
                    else:
                        await queue.put(SyncTask(
                            self._source_type, f["id"], "upsert",
                            data_source_id=self._data_source_id,
                        ))
                        n_upserts += 1

                page_token = response.get("nextPageToken")
                if not page_token:
                    break

        # Persist the full folder ID set so the incremental path can filter
        # Changes API results (which are Drive-wide) to our subtree.
        manifest.set_source_state(
            self._data_source_id, _STATE_FOLDER_IDS,
            json.dumps(sorted(all_folder_ids)),
        )

        # Anchor the Changes cursor at "now" for upcoming _watch calls.
        token_resp = await asyncio.to_thread(
            service.changes().getStartPageToken().execute
        )
        manifest.set_source_state(
            self._data_source_id, _STATE_PAGE_TOKEN, token_resp["startPageToken"]
        )

        return n_upserts

    async def _incremental_scan(
        self,
        service,
        queue: asyncio.Queue,
        manifest: "Manifest",
        stored_token: str,
    ) -> tuple[int, int]:
        """Drain Changes API since stored_token. Returns (n_upserts, n_deletes)."""
        con = manifest.connect()
        try:
            known_refs = manifest.get_source_refs(con, self._data_source_id)
        finally:
            con.close()

        folder_ids_str = manifest.get_source_state(self._data_source_id, _STATE_FOLDER_IDS)
        watched_folder_ids: set[str] = (
            set(json.loads(folder_ids_str)) if folder_ids_str else {self._folder_id}
        )

        n_upserts = n_deletes = 0
        page_token = stored_token

        while page_token:
            response = await asyncio.to_thread(
                service.changes().list(
                    pageToken=page_token,
                    spaces="drive",
                    fields=(
                        "nextPageToken,newStartPageToken,"
                        "changes(fileId,removed,file(id,parents,trashed,mimeType))"
                    ),
                    pageSize=100,
                ).execute
            )

            folder_ids_updated = False
            for change in response.get("changes", []):
                file_id = change["fileId"]
                file_info = change.get("file") or {}
                # When removed=true the file object is often absent, so mimeType
                # is unavailable. Use set membership to distinguish files from folders.
                removed = change.get("removed") or file_info.get("trashed", False)

                if removed:
                    if file_id in known_refs:
                        await queue.put(SyncTask(
                            self._source_type, file_id, "delete",
                            data_source_id=self._data_source_id,
                        ))
                        n_deletes += 1
                    if file_id in watched_folder_ids:
                        # Drive sends individual removal events for each item
                        # inside a deleted folder, so just clean up the folder set.
                        watched_folder_ids.discard(file_id)
                        folder_ids_updated = True
                else:
                    mime = file_info.get("mimeType", "")
                    parents = set(file_info.get("parents") or [])
                    in_tree = bool(parents & watched_folder_ids)

                    if mime == _FOLDER_MIME:
                        # New subfolder appeared in our tree — register it immediately
                        # so files added to it later on this same page pass in_tree.
                        if in_tree and file_id not in watched_folder_ids:
                            watched_folder_ids.add(file_id)
                            folder_ids_updated = True
                    else:
                        if file_id in known_refs:
                            if in_tree:
                                await queue.put(SyncTask(
                                    self._source_type, file_id, "upsert",
                                    data_source_id=self._data_source_id,
                                ))
                                n_upserts += 1
                            else:
                                # File we were tracking moved out of our watched tree.
                                await queue.put(SyncTask(
                                    self._source_type, file_id, "delete",
                                    data_source_id=self._data_source_id,
                                ))
                                n_deletes += 1
                        elif in_tree:
                            await queue.put(SyncTask(
                                self._source_type, file_id, "upsert",
                                data_source_id=self._data_source_id,
                            ))
                            n_upserts += 1

            if folder_ids_updated:
                manifest.set_source_state(
                    self._data_source_id, _STATE_FOLDER_IDS,
                    json.dumps(sorted(watched_folder_ids)),
                )

            if "newStartPageToken" in response:
                manifest.set_source_state(
                    self._data_source_id, _STATE_PAGE_TOKEN,
                    response["newStartPageToken"],
                )
                break
            page_token = response.get("nextPageToken")

        return n_upserts, n_deletes

    async def _catch_up(
        self,
        queue: asyncio.Queue,
        manifest: "Manifest",
        force_reindex: bool = False,
    ) -> None:
        service = await asyncio.to_thread(self._build_service)
        stored_token = manifest.get_source_state(self._data_source_id, _STATE_PAGE_TOKEN)

        if force_reindex or stored_token is None:
            n_upserts = await self._full_scan(service, queue, manifest)
            n_deletes = 0
        else:
            n_upserts, n_deletes = await self._incremental_scan(
                service, queue, manifest, stored_token
            )

        log.info(
            "Catch-up complete — %d upsert(s)  %d delete(s)",
            n_upserts, n_deletes,
        )

    async def _watch(
        self,
        queue: asyncio.Queue,
        manifest: "Manifest",
        catchup_done: asyncio.Event,
    ) -> None:
        from googleapiclient.errors import HttpError

        await catchup_done.wait()
        service = await asyncio.to_thread(self._build_service)
        while True:
            await asyncio.sleep(self._poll_interval)
            stored_token = manifest.get_source_state(self._data_source_id, _STATE_PAGE_TOKEN)
            try:
                n_upserts, n_deletes = await self._incremental_scan(
                    service, queue, manifest, stored_token
                )
            except HttpError as exc:
                if exc.resp.status != 401:
                    raise
                log.warning("Google Drive access token expired — refreshing.")
                service = await asyncio.to_thread(self._build_service)
                n_upserts, n_deletes = await self._incremental_scan(
                    service, queue, manifest, stored_token
                )
            if n_upserts or n_deletes:
                log.info(
                    "Changes detected — %d upsert(s)  %d delete(s)",
                    n_upserts, n_deletes,
                )
