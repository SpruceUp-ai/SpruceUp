# SpruceUp

MVP:
SpruceUp keeps a RAG application's Postgres vector table in sync with a local file corpus. When files are added, modified, moved, or deleted, it re-chunks and re-embeds only the affected files and reconciles the diff into Postgres. An SQLite manifest tracks file state between runs so restarts are incremental.

FUTURE PLANS:
SpruceUp keeps a RAG application's vector db table in sync with their data corpus, offering support for a number of different vector db providers as well as file hosting platforms. When files are added, modified, moved, or deleted, it re-chunks and re-embeds only the affected files and reconciles the diffs into the user's target db, letting the user connect to a variety of different embedding models. The diffing processing makes heavy use of memoization to efficiently re-processed file changes in large corpuses. An SQLite manifest tracks file state between runs so restarts are incremental.

## Running the project

```bash
# Install dependencies
poetry install

# Run (from the project root, with PG_CONNSTR set in the environment or .env)
spruceup start
# or, without activating the venv:
python -m spruceup start

# Tests (Postgres is mocked; SQLite runs against a real temp file)
poetry run pytest
```

Requires `PG_CONNSTR` and `OPENAI_API_KEY` to be set in the environment (or loaded via `python-dotenv` inside `spruceup_pipeline.py`) and PostgreSQL running locally with the `pgvector` extension installed.

## Architecture and data flow

```
spruceup start
  └─ spruceup/cli.py          → adds CWD to sys.path, imports spruceup_pipeline
  └─ pipeline_validator.py    → validates config is SpruceUpConfig + @transform registered
  └─ spruceup/app.py          → wires all components together, starts the event loop
       └─ Monitor.run()       → LocalFileWatcher._catch_up() scans dir → SyncTask on queue
       └─ LocalFileWatcher._watch() → listens for live changes via watchfiles.awatch
       └─ Coordinator.run()   → pulls SyncTask → fetches file → @transform(embed=...)
                               → validate_schema_objects() → SyncEngine.reconcile()
       └─ SyncEngine.reconcile() → diffs chunks → upserts/deletes Postgres + SQLite manifest
```

Move events are handled by `SyncEngine.move_file()`, which updates only the SQLite manifest (Postgres vectors use content-based PKs and remain valid after a rename). Delete and upsert events go through the full `Coordinator.process_task()` pipeline.

## File map

| Path | Role |
|------|------|
| `spruceup_pipeline.py` | User-defined pipeline: schema dataclass, `@transform` function, and `defineConfig()` call |
| `spruceup/cli.py` | `spruceup start` entry point; discovers and imports the pipeline file |
| `spruceup/app.py` | Async `run(pipeline)` function; wires all components and starts the event loop |
| `spruceup/pipeline_validator.py` | Validates `config` is a `SpruceUpConfig` and `@transform` is registered at startup |
| `spruceup/config.py` | `SpruceUpConfig` dataclass + `defineConfig()` — validates and captures pipeline config |
| `spruceup/connectors/base.py` | Abstract base classes: `SourceConnector`, `TargetConnector`, `EmbedderConfig` |
| `spruceup/connectors/sources/local.py` | `LocalFilesSource` — watches a local directory |
| `spruceup/connectors/targets/pgvector.py` | `PgVectorTarget` — writes to a Postgres pgvector table |
| `spruceup/connectors/embedders/openai.py` | `OpenAIEmbedder` — embeds via OpenAI API |
| `spruceup/registry.py` | `@transform` decorator; singleton `TransformTracker` |
| `spruceup/manifest.py` | `Manifest` class — all SQLite manifest reads and writes, including transform hash management |
| `spruceup/models.py` | Core dataclasses: `SpruceFile`, `ChunkWrapper`, `TargetTableConfig`, `UserDefinedChunkSchema` |
| `spruceup/validation.py` | `validate_schema_objects()` — checks transform output against declared schema |
| `spruceup/hashing.py` | All hashing functions (BLAKE2B, 16-byte digests throughout) |
| `spruceup/db.py` | SQLite schema init (`init_db`) |
| `spruceup/coordinator.py` | `Coordinator` — drives the per-file pipeline; `LocalFileFetcher` |
| `spruceup/embedding.py` | `Embedder` + `OpenAIProvider` — batched, concurrent, retried via tenacity |
| `spruceup/monitoring/monitor.py` | `Monitor`, `LocalFileWatcher` (`_catch_up` + `_watch`), `_BufferedQueue` |
| `spruceup/monitoring/tasks.py` | `SyncTask` dataclass |
| `spruceup/monitoring/capture.py` | `TransformTracker` — registers transform functions, exposes their hashes |
| `spruceup/sync_engine/sync_engine.py` | `SyncEngine.reconcile()`, `delete_file()`, `move_file()` |
| `spruceup/sync_engine/pgvector.py` | Postgres read/write functions |
| `example/` | Example chunking logic consumed by `spruceup_pipeline.py` |
| `tests/test_sync_engine.py` | Unit tests for `SyncEngine` |
| `tests/test_validation.py` | Unit tests for `validate_schema_objects` |

## SQLite manifest schema

```sql
data_sources  (id INTEGER PK AUTOINCREMENT, source_type TEXT)
files         (id BLOB PK,                  -- hash_file_path(file_path)
               file_path TEXT NOT NULL,
               inode INTEGER,               -- used by monitor for move detection
               content_hash BLOB, mtime REAL, data_source_id INTEGER FK, file_type TEXT)
chunks        (id BLOB PK,                  -- hash_chunk_id(file_path, ordinal)
               file_id BLOB FK,
               user_chunk_object_hash BLOB, -- change detection
               user_chunk_object BLOB)      -- JSON-serialized user dataclass
transform_hashes (transform_hash BLOB PK)  -- hash is PK, not func_name
```

`transform_hashes` stores the BLAKE2B hash of each transform function's source. Using the hash (not the function name) as PK means renaming a function doesn't trigger a false full-reindex.

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

**`@transform`** — async; receives `file_props` and an `embed` callable, returns a list of your schema objects:
```python
from spruceup import transform

@transform
async def my_transform(*, file_props: dict, embed) -> list[MyChunk]:
    # file_props keys: raw_content, file_path, mtime, file_type
    chunks = split_into_chunks(file_props["raw_content"])
    embeddings = await embed(chunks)  # returns list[list[float]]
    return [MyChunk(id=..., chunk_text=c, chunk_embedding=e) for c, e in zip(chunks, embeddings)]
```

**`defineConfig`** — wire together your source(s), target, and embedding provider:
```python
import os
from spruceup import defineConfig, LocalFilesSource, PgVectorTarget, OpenAIEmbedder

config = defineConfig(
    sources=[
        LocalFilesSource(watched_dir="path/to/corpus"),
    ],
    target=PgVectorTarget(
        connstr=os.environ["PG_CONNSTR"],  # load secrets from env, not hardcoded
        table="my_table",
        schema=MyChunk,
        primary_key="id",
    ),
    embeddings=OpenAIEmbedder(
        model="text-embedding-3-small",
    ),
)
```

## Key invariants

- **All IDs are BLAKE2B 16-byte digests** — file IDs, chunk IDs, content hashes, object hashes, transform hashes all use `hashing.py` with `digest_size=16`.
- **`conn.executemany()` does not exist in psycopg3** — always use `with conn.cursor() as cur: cur.executemany(...)`.
- **`_BufferedQueue`** — captures `_watch` events that arrive while `_catch_up` is still running, then replays them in order once catch-up is complete. This prevents double-processing a file that changed between startup scan and watch start.
- **`_watch` filters directories** — `pathlib.Path(path).is_file()` guard is required because `watchfiles` can emit events for the watched directory itself when files are added to it.
- **Postgres vectors survive moves** — `SyncEngine.move_file()` updates only the SQLite manifest; no Postgres writes happen. Chunk PKs are content-based (user-defined), not path-based.
- **`ensure_file_row_exists` before chunk writes** — the `files` FK on `chunks` requires the file row to exist before any chunk for that file is inserted.
- **`Manifest` is the sole SQLite access point** — all reads and writes to `spruceup_manifest.db` go through the `Manifest` class; methods that need to be atomic share a connection via `manifest.connect()` used as a context manager.
- **Pipeline validation runs before anything starts** — `pipeline_validator.py` checks that `config` is a `SpruceUpConfig` and `@transform` is registered before `init_db` or any network connections are made. Field-level validation (non-empty strings, correct types) happens eagerly inside `defineConfig()` at import time.
