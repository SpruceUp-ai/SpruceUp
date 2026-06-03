# SpruceUp

MVP:
SpruceUp keeps a RAG application's Postgres vector table in sync with a local file corpus. When files are added, modified, moved, or deleted, it re-chunks and re-embeds only the affected files and reconciles the diff into Postgres. An SQLite manifest tracks file state between runs so restarts are incremental, and an `@memoize` decorator caches expensive per-file subfunction results across runs.

FUTURE PLANS:
SpruceUp keeps a RAG application's vector db table in sync with their data corpus, offering support for a number of different vector db providers as well as file hosting platforms. When files are added, modified, moved, or deleted, it re-chunks and re-embeds only the affected files and reconciles the diffs into the user's target db, letting the user connect to a variety of different embedding models. The diffing process makes heavy use of memoization to efficiently re-process file changes in large corpuses. An SQLite manifest tracks file state between runs so restarts are incremental.

## Running the project

```bash
# Install dependencies (Python 3.14+)
poetry install

# Run (from the project root, with PG_CONNSTR and OPENAI_API_KEY set in env or .env)
spruceup start
# or, without activating the venv:
python -m spruceup start

# Tests (Postgres is mocked; SQLite runs against a real temp file)
poetry run pytest
```

Requires `PG_CONNSTR` and `OPENAI_API_KEY` to be set in the environment (or loaded via `python-dotenv`) and PostgreSQL running locally with the `pgvector` extension installed.

## Architecture and data flow

```
spruceup start
  └─ spruceup/cli.py                  → adds CWD to sys.path, imports spruceup_pipeline
  └─ utils/validation.validate_pipeline → confirms config is SpruceUpConfig with a transform
  └─ spruceup/app.py                  → wires all components together, starts the event loop
       └─ Manifest()                  → opens/creates spruceup_manifest.db (auto-init)
       └─ hash_transform              → detects transform changes → force_reindex flag
       └─ target.ensure_table_exists  → creates the target table if missing
       └─ manifest.register_source    → upsert one data_sources row per configured source
       └─ sync_engine.delete_stale_sources → drops chunks for sources removed from config
       └─ Monitor.run() per source    → LocalFileWatcher._catch_up scans dir → SyncTask on queue
                                       LocalFileWatcher._watch listens via watchfiles.awatch
       └─ Coordinator.run()           → pulls SyncTask → source.fetch → user transform
                                       → validate_schema_objects → SyncEngine.reconcile()
       └─ SyncEngine.reconcile()      → diffs chunks → upserts/deletes Postgres + SQLite manifest
```

Move events are handled by `SyncEngine.move_file()`, which only touches the SQLite manifest — Postgres rows use content-derived primary keys and survive a rename. Delete and upsert events go through the full coordinator pipeline.

## File map

| Path | Role |
|------|------|
| [spruceup_pipeline.py](spruceup_pipeline.py) | User-defined pipeline: schema dataclass, async `transform` function, and `defineConfig()` call |
| [spruceup/cli.py](spruceup/cli.py) | `spruceup start` entry point; discovers and imports the pipeline file |
| [spruceup/__main__.py](spruceup/__main__.py) | Enables `python -m spruceup start` |
| [spruceup/app.py](spruceup/app.py) | Async `run(pipeline)` function; wires all components and starts the event loop |
| [spruceup/config.py](spruceup/config.py) | `SpruceUpConfig` dataclass + `defineConfig()` — validates and captures pipeline config |
| [spruceup/coordinator.py](spruceup/coordinator.py) | `Coordinator` — drives the per-file pipeline; calls source → transform → sync_engine |
| [spruceup/manifest.py](spruceup/manifest.py) | `Manifest` class — all SQLite manifest reads/writes; owns schema init for files, chunks, data_sources, transform_hashes, memoize_cache |
| [spruceup/models.py](spruceup/models.py) | Core dataclasses: `SpruceFile`, `ChunkWrapper`, `SyncTask` |
| [spruceup/connectors/base.py](spruceup/connectors/base.py) | Abstract bases: `SourceConnector`, `TargetConnector`, `EmbedderConnector` |
| [spruceup/connectors/sources/local.py](spruceup/connectors/sources/local.py) | `LocalFilesSource` — watches a local directory, fetches file bytes |
| [spruceup/connectors/sources/google_drive.py](spruceup/connectors/sources/google_drive.py) | `GoogleDriveSource` — watches a Google Drive folder, fetches file bytes (stub) |
| [spruceup/connectors/targets/pgvector.py](spruceup/connectors/targets/pgvector.py) | `PgVectorTarget` — writes to a Postgres pgvector table, including `ensure_table_exists` schema introspection |
| [spruceup/connectors/embedders/openai.py](spruceup/connectors/embedders/openai.py) | `OpenAIEmbedder` — calls the OpenAI embeddings API with tenacity retry |
| [spruceup/connectors/embedders/embedding_batcher.py](spruceup/connectors/embedders/embedding_batcher.py) | `EmbeddingBatcher` — wraps an inner embedder and merges chunks across concurrent files into batched API calls |
| [spruceup/sync_engine/sync_engine.py](spruceup/sync_engine/sync_engine.py) | `SyncEngine.reconcile()`, `delete_file()`, `move_file()`, `delete_stale_sources()` |
| [spruceup/monitoring/monitor.py](spruceup/monitoring/monitor.py) | `Monitor`, `BaseWatcher` (template `run`, abstract `_catch_up`+`_watch`), `_BufferedQueue`, `_with_retry` |
| [spruceup/monitoring/local_file_watcher.py](spruceup/monitoring/local_file_watcher.py) | `LocalFileWatcher` — implements `_catch_up` + `_watch` for local filesystem sources |
| [spruceup/monitoring/google_drive_watcher.py](spruceup/monitoring/google_drive_watcher.py) | `GoogleDriveWatcher` — implements `_catch_up` + `_watch` for Google Drive sources (stub) |
| [spruceup/memoize/decorator.py](spruceup/memoize/decorator.py) | `@memoize(returns=...)` decorator for caching subfunction results per file |
| [spruceup/memoize/context.py](spruceup/memoize/context.py) | ContextVars holding the active manifest, file_id, and temp_keys set used by the decorator |
| [spruceup/memoize/serialization.py](spruceup/memoize/serialization.py) | `validate_return_type`, `serialize`, `deserialize` for memoize cache values |
| [spruceup/utils/hashing.py](spruceup/utils/hashing.py) | All hashing functions (BLAKE2B, 16-byte digests throughout), including `hash_args` for memoize |
| [spruceup/utils/validation.py](spruceup/utils/validation.py) | `validate_schema_objects` (transform output) + `validate_pipeline` (startup check) |
| [example/](example/) | Example chunking helpers consumed by `spruceup_pipeline.py` |
| [tests/test_sync_engine.py](tests/test_sync_engine.py) | Unit tests for `SyncEngine` (reconcile, delete_file, move_file, delete_stale_sources) |
| [tests/test_memoize.py](tests/test_memoize.py) | Unit tests for `@memoize` and the memoize_cache table |
| [tests/test_embedding_batcher.py](tests/test_embedding_batcher.py) | Unit tests for `EmbeddingBatcher` cross-file batching |
| [tests/test_monitor_retry.py](tests/test_monitor_retry.py) | Unit tests for `_with_retry` watcher recovery |
| [tests/test_validation.py](tests/test_validation.py) | Unit tests for `validate_schema_objects` |

## SQLite manifest schema

```sql
data_sources    (id INTEGER PK AUTOINCREMENT,
                 source_type TEXT NOT NULL,
                 source_identifier TEXT NOT NULL,
                 UNIQUE(source_type, source_identifier))

source_state    (data_source_id INTEGER REFERENCES data_sources(id) ON DELETE CASCADE,
                 key TEXT NOT NULL,
                 value TEXT NOT NULL,
                 PRIMARY KEY (data_source_id, key))
                 -- connector-specific cursor state (e.g. Google Drive page tokens)

files           (id TEXT PRIMARY KEY,         -- local: "{inode}:{path}"; Drive: Drive file ID
                 content_hash BLOB,
                 data_source_id INTEGER REFERENCES data_sources(id) ON DELETE CASCADE,
                 file_type TEXT,
                 raw_content BLOB,
                 modified_at REAL,            -- Unix epoch float; set by every source connector
                 sync_state TEXT NOT NULL DEFAULT 'in_flight')

chunks          (file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE ON UPDATE CASCADE,
                 user_chunk_object_hash BLOB NOT NULL,
                 PRIMARY KEY (file_id, user_chunk_object_hash))

transform_hashes (transform_hash BLOB PRIMARY KEY)  -- hash is PK, not func_name

memoize_fn_hashes (fn_hash BLOB PRIMARY KEY)

memoize_cache   (file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE ON UPDATE CASCADE,
                 fn_hash BLOB NOT NULL,
                 args_hash BLOB NOT NULL,
                 result BLOB NOT NULL,
                 PRIMARY KEY (file_id, fn_hash, args_hash))

embedding_cache (file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE ON UPDATE CASCADE,
                 chunk_text_hash BLOB NOT NULL,
                 embedding BLOB NOT NULL,
                 PRIMARY KEY (file_id, chunk_text_hash))

config_state    (key TEXT PRIMARY KEY,
                 value TEXT NOT NULL)
```

`transform_hashes` stores the BLAKE2B hash of the transform function's source. Using the hash (not the function name) as PK means renaming a function doesn't trigger a false full-reindex.

`source_state` stores per-source connector cursors that persist across restarts (e.g. Google Drive Changes API page tokens, webhook channel expiry times).

`memoize_cache` and `embedding_cache` rows live and die with their owning file via `ON DELETE CASCADE`. `ON UPDATE CASCADE` propagates a `file_id` rename (from `update_file_id`) without breaking child rows, so caches survive a rename.

## Customising the pipeline

Edit `spruceup_pipeline.py`. The required pieces are:

**Schema dataclass** — define the fields your Postgres table will have:
```python
from dataclasses import dataclass

@dataclass
class MyChunk:
    id: str               # primary key
    chunk_text: str       # text used for embedding
    chunk_embedding: list[float]
    my_custom_field: str  # any additional metadata columns
```

**Transform function** — async; receives `file_props` and an `embed` callable, returns a list of your schema objects. It is passed directly to `defineConfig` — there is no decorator:
```python
async def my_transform(*, file_props: FileProps, embed) -> list[MyChunk]:
    # file_props fields: raw_content, display_name, file_type, modified_at
    chunks = split_into_chunks(file_props.raw_content)
    embeddings = await embed(chunks)  # returns list[list[float]]
    return [MyChunk(id=..., chunk_text=c, chunk_embedding=e) for c, e in zip(chunks, embeddings)]
```

**`defineConfig`** — wire together your source(s), target, embedder, and transform:
```python
import os
from spruceup import defineConfig, LocalFilesSource, PgVectorTarget, OpenAIEmbedder

config = defineConfig(
    sources=[
        LocalFilesSource(watched_dir="path/to/corpus"),
        # LocalFilesSource(watched_dir="path/to/another_corpus"),
    ],
    target=PgVectorTarget(
        connstr=os.getenv("PG_CONNSTR"),  # load secrets from env, not hardcoded
        table="my_table",
        schema=MyChunk,
        primary_key="id",
    ),
    embedder=OpenAIEmbedder(
        model="text-embedding-3-small",
    ),
    transform=my_transform,
)
```

**Optional `@memoize`** — cache expensive subfunction results per file across runs. Only valid when called from within the transform function:
```python
from spruceup import memoize

@memoize(returns=list)
def extract_sections(raw_text: str) -> list[str]:
    ...  # expensive parse; cached in memoize_cache and reused on unchanged files

async def my_transform(*, file_props, embed):
    sections = extract_sections(file_props["raw_content"])
    ...
```
Supported `returns` types: `str, int, float, bool, list, dict`. The cache key is `(file_id, fn_hash, args_hash)`. Entries not touched during a transform run are swept after the run completes (`Manifest.sweep_memoized`), so stale args don't accumulate.

## Key invariants

- **File IDs are opaque strings, not hashes** — local: `f"{inode}:{path}"`; Drive: the Drive file ID string. Content hashes, chunk object hashes, transform hashes, and memoize fn/args hashes are all BLAKE2B 16-byte digests from `utils/hashing.py`.
- **Transform is passed in, not decorated** — there is no `@transform` decorator. `defineConfig(transform=fn)` is the only registration path.
- **`Manifest` is the sole SQLite access point** — all reads/writes to `spruceup_manifest.db` go through the `Manifest` class. Methods that need to be atomic share a connection via `manifest.connect()` used as a context manager; standalone helpers manage their own connections internally.
- **Manifest auto-initializes its schema** — constructing `Manifest()` calls `_init_db()`, so no separate `init_db` step exists.
- **`_BufferedQueue`** — captures `_watch` events that arrive while `_catch_up` is still running, then replays them in order once catch-up is complete. This prevents double-processing a file that changed between startup scan and watch start.
- **`_watch` filters directories** — `pathlib.Path(path).is_file()` guard is required because `watchfiles` can emit events for the watched directory itself when files are added to it.
- **`file_id` encodes connector identity** — for local files, `f"{inode}:{path}"` makes the inode the stable key so renames are detected without re-hashing. For Drive, the Drive file ID is used directly. `SyncTask.identifier` carries the human-readable path or Drive ID used for fetch and display; `SyncTask.current_file_id` carries the stable `file_id` before the action.
- **Postgres vectors survive moves** — `SyncEngine.move_file()` updates only the SQLite manifest; no Postgres writes happen. Chunk PKs are content-derived (user-defined), not path-based.
- **`ensure_file_row_exists` before chunk writes** — the `files` FK on `chunks` requires the file row to exist before any chunk for that file is inserted; `SyncEngine.reconcile` calls it explicitly before chunk upserts.
- **`upsert_file_row` uses `ON CONFLICT DO UPDATE`** — in-place update, not DELETE+INSERT, so it does not trigger the `ON DELETE CASCADE` on `chunks.file_id` or `memoize_cache.file_id`.
- **Pipeline validation runs before anything starts** — `utils/validation.validate_pipeline` checks that `config` is a `SpruceUpConfig` with a registered transform before any DB or network connection. Field-level validation (types, non-empty lists) happens eagerly inside `defineConfig()` at import time.
- **Multi-source bookkeeping** — each `defineConfig` source becomes a row in `data_sources` (`source_type` + `source_identifier`, unique). `SyncEngine.delete_stale_sources(active_ids)` at startup deletes target rows and manifest rows for any source no longer in the config.
- **`@memoize` requires transform context** — the decorator reads `_memo_manifest_var`, `_memo_file_id_var`, and `_memo_temp_keys_var` ContextVars set by `Coordinator.upsert_file`. Calling a memoized function outside that scope raises `RuntimeError`.
- **`conn.executemany()` does not exist in psycopg3** — always use `with conn.cursor() as cur: cur.executemany(...)` (see `PgVectorTarget.sync`).
- **Watcher retry wraps startup only** — `_with_retry` retries `watcher.run()` with exponential backoff up to 20 attempts. A healthy watcher's own loop never returns, so the retry logic never re-enters on a running watcher.
- **A Drive `service` must never have two requests in flight at once** — `build("drive", "v3", ...)` holds one `httplib2.Http` with a single, unlocked socket; two concurrent `to_thread(...execute)` calls on the same service race on that socket and corrupt responses. Current code is safe because reuse is *sequential*: `GoogleDriveSource.fetch` builds a fresh service per call, and `GoogleDriveWatcher` awaits its calls one at a time. Do NOT cache the service on the instance and then fan out concurrent `fetch()`es. If a shared long-lived client is ever needed, swap the transport to `AuthorizedSession` from `google.auth.transport.requests` (urllib3 gives each thread its own connection) — not a lock. `build()` (google-api-python-client ≥ 2.0) uses bundled static discovery, so per-call builds are cheap.
