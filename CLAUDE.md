# SpruceUp

MVP:
SpruceUp keeps a RAG application's Postgres vector table in sync with a local file corpus. When files are added, modified, moved, or deleted, it re-chunks and re-embeds only the affected files and reconciles the diff into Postgres. An SQLite manifest tracks file state between runs so restarts are incremental.

FUTURE PLANS:
SpruceUp keeps a RAG application's vector db table in sync with their data corpus, offering support for a number of different vector db providers as well as file hosting platforms. When files are added, modified, moved, or deleted, it re-chunks and re-embeds only the affected files and reconciles the diffs into the user's target db, letting the user connect to a variety of different embedding models. The diffing processing makes heavy use of memoization to efficiently re-processed file changes in large corpuses. An SQLite manifest tracks file state between runs so restarts are incremental.

## Running the project

```bash
# First run
poetry install
poetry run python main.py

# Tests (Postgres is mocked; SQLite runs against a real temp file)
poetry run pytest
```

Requires a `.env` file with `OPENAI_API_KEY=...` and PostgreSQL running locally at `postgresql://localhost:5432/spruce_lecture_rag` with the `pgvector` extension installed.

## Architecture and data flow

```
main.py
  └─ imports spruceup_pipeline   → @file_transform / @chunk_transform decorators fire
  └─ Monitor.run()               → LocalFileWatcher._catch_up() scans dir → SyncTask on queue
  └─ LocalFileWatcher._watch()   → listens for live changes via watchfiles.awatch
  └─ Coordinator.run()           → pulls SyncTask → fetches file → file_transform
                                   → chunk_transform(embed=...) → SyncEngine.reconcile()
  └─ SyncEngine.reconcile()      → diffs chunks → upserts/deletes Postgres + SQLite manifest
```

Move events are handled by `SyncEngine.move_file()`, which updates only the SQLite manifest (Postgres vectors use content-based PKs and remain valid after a rename). Delete and upsert events go through the full `Coordinator.process_task()` pipeline.

## File map

| Path | Role |
|------|------|
| `main.py` | Entry point; wires all components together |
| `spruceup_pipeline.py` | User-defined pipeline (transforms + config constants) |
| `spruceup/registry.py` | `@file_transform` / `@chunk_transform` decorators; singleton `TransformTracker` |
| `spruceup/models.py` | Core dataclasses: `SpruceFile`, `ChunkWrapper`, `TargetTableConfig`, `UserDefinedChunkSchema` |
| `spruceup/hashing.py` | All hashing functions (BLAKE2B, 16-byte digests throughout) |
| `spruceup/db.py` | SQLite schema init (`init_db`) |
| `spruceup/coordinator.py` | `Coordinator` — drives the per-file pipeline; `LocalFileFetcher` |
| `spruceup/embedding.py` | `Embedder` + `OpenAIProvider` — batched, concurrent, retried via tenacity |
| `spruceup/monitoring/monitor.py` | `Monitor`, `LocalFileWatcher` (`_catch_up` + `_watch`), `_BufferedQueue` |
| `spruceup/monitoring/tasks.py` | `SyncTask` dataclass |
| `spruceup/monitoring/capture.py` | `TransformTracker` — detects whether transform functions changed |
| `spruceup/sync_engine/sync_engine.py` | `SyncEngine.reconcile()`, `delete_file()`, `move_file()` |
| `spruceup/sync_engine/manifest.py` | SQLite manifest read/write functions |
| `spruceup/sync_engine/target_db.py` | Postgres read/write functions |
| `example/` | Example chunking logic consumed by `spruceup_pipeline.py` |
| `tests/test_sync_engine.py` | Unit tests for `SyncEngine` |

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

Edit `spruceup_pipeline.py`. The two required pieces are:

**`@file_transform`** — synchronous; receives a `file_props` dict and returns `list[str]` (the chunk strings that will be embedded):
```python
@file_transform
def my_file_transform(*, file_props: dict) -> list[str]:
    # keys: raw_content, file_path, mtime, file_type
    ...
```

**`@chunk_transform`** — async; receives the chunk strings plus an `embed` callable and returns a list of your schema dataclass objects:
```python
@chunk_transform
async def my_chunk_transform(chunk_strs: list[str], *, embed) -> list[MyChunk]:
    embeddings = await embed(chunk_strs)  # returns list[list[float]]
    ...
```

The framework never touches `chunk_embedding` directly. The user calls `embed` and assigns the result wherever they want in their schema object.

Also set these constants at the bottom of `spruceup_pipeline.py`:
```python
CHUNK_SCHEMA = MyChunk      # the dataclass class itself
TARGET_DB    = "..."        # Postgres database name
TARGET_TABLE = "..."        # table name to upsert into
PRIMARY_KEY  = "id"         # field on MyChunk used as the Postgres PK
WATCHED_DIR  = "path/to/corpus"
```

## Key invariants

- **All IDs are BLAKE2B 16-byte digests** — file IDs, chunk IDs, content hashes, object hashes, transform hashes all use `hashing.py` with `digest_size=16`.
- **`conn.executemany()` does not exist in psycopg3** — always use `with conn.cursor() as cur: cur.executemany(...)`.
- **`_BufferedQueue`** — captures `_watch` events that arrive while `_catch_up` is still running, then replays them in order once catch-up is complete. This prevents double-processing a file that changed between startup scan and watch start.
- **`_watch` filters directories** — `pathlib.Path(path).is_file()` guard is required because `watchfiles` can emit events for the watched directory itself when files are added to it.
- **Postgres vectors survive moves** — `SyncEngine.move_file()` updates only the SQLite manifest; no Postgres writes happen. Chunk PKs are content-based (user-defined), not path-based.
- **`ensure_file_row_exists` before chunk writes** — the `files` FK on `chunks` requires the file row to exist before any chunk for that file is inserted.
