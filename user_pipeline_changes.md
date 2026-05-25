# User Pipeline Changes — Architecture Overview

## What Changed and Why

### The Problem with Loose Constants

Before these changes, users configured SpruceUp by defining five module-level constants in their `spruceup_pipeline.py`:

```python
CHUNK_SCHEMA = LectureChunk
TARGET_TABLE = "data_chunks"
PRIMARY_KEY  = "id"
WATCHED_DIR  = "example/data_corpus"
PG_CONNSTR   = os.environ["PG_CONNSTR"]
```

This had several problems:
- **No structure** — five unrelated names floating at module scope with no grouping
- **Late errors** — a misspelled or missing constant was only caught at startup, deep inside `app.py`
- **Not extensible** — adding a second watched directory, a different target database, or a different embedding provider required inventing new constant names with no clear convention
- **Hardcoded internals** — `app.py` was tightly coupled to specific implementations: it imported `OpenAIProvider` directly, called `LocalFileWatcher` by name, and passed `pg_connstr` directly to `SyncEngine`

### The Solution: `defineConfig` + Connector ABCs

The new API replaces the five constants with a single structured call:

```python
from spruceup import defineConfig, transform
from spruceup import LocalFilesSource, PgVectorTarget, OpenAIEmbedder

config = defineConfig(
    sources=[
        LocalFilesSource(watched_dir="example/data_corpus"),
    ],
    target=PgVectorTarget(
        connstr=os.environ["PG_CONNSTR"],
        table="data_chunks",
        schema=LectureChunk,
        primary_key="id",
    ),
    embeddings=OpenAIEmbedder(
        model="text-embedding-3-small",
    ),
)
```

`defineConfig` validates all required fields eagerly — at the moment the module is imported — rather than at startup. Errors surface immediately with a clear message.

Adding a second watched directory is now a natural API operation, not an ad-hoc convention:

```python
config = defineConfig(
    sources=[
        LocalFilesSource(watched_dir="corpus/lectures"),
        LocalFilesSource(watched_dir="corpus/readings"),
    ],
    ...
)
```

### The Problem with a Single Shared Data Source

SpruceUp's SQLite manifest tracks which files it has seen and what state they were in. Previously, the `data_sources` table held exactly one hardcoded row — `id=1, source_type='local'` — inserted unconditionally at startup. All file rows in the manifest were stamped with `data_source_id=1`, regardless of which directory they came from.

This worked when there was only one watched directory. With multiple sources, it fails silently: both `LocalFilesSource` watchers would scan the entire `files` table during catch-up, meaning each watcher would see the other source's files and try to delete them when it didn't find them on disk.

### The Fix: Per-Instance Data Sources

Each configured source now gets its own row in `data_sources`, identified by its type and a stable `source_identifier` (for `LocalFilesSource`, the normalized absolute path via `pathlib.Path.resolve()`):

```sql
data_sources (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type       TEXT NOT NULL,
    source_identifier TEXT NOT NULL,
    UNIQUE(source_type, source_identifier)
)
```

At startup, `manifest.register_source(source_type, source_identifier)` does an idempotent `INSERT OR IGNORE` followed by a `SELECT id` — so the same source always resolves to the same database row across restarts. The returned `id` flows into the watcher and into every `SyncTask` that watcher produces, scoping all manifest queries for that watcher to its own files.

Sources that appear in the database but are no longer in the user's `config.sources` list are deleted at startup. Because `files.data_source_id` references `data_sources.id` with `ON DELETE CASCADE`, and `chunks.file_id` references `files.id` with `ON DELETE CASCADE`, this cascades automatically — removing a stale source row removes all its file rows, which removes all their chunk rows, in one SQL statement.

---

## Why the Architecture Changed (Not Just the API)

### `app.py` Was the Integration Point

Before these changes, `app.py` hardcoded every specific implementation:

```python
# app.py (before)
sync_engine = SyncEngine(manifest=manifest, pg_connstr=pipeline.PG_CONNSTR)
embedder = Embedder(provider=OpenAIProvider())                     # hardcoded
monitor.add_watcher(LocalFileWatcher(pipeline.WATCHED_DIR))        # hardcoded
```

If we had simply introduced `defineConfig` but left `app.py` unchanged, it would have read `config.target.connstr` and still called `PgVectorSyncTarget(...)` by name, and still imported `OpenAIProvider` directly. The config would look structured from the user's side, but `app.py` would still be a hardcoded dispatch table. Adding a second target type (e.g., Pinecone) would require editing `app.py` with `if isinstance(target, PineconeTarget): ...` branches.

### The Fix: Factory Methods on the Connectors

Each connector now knows how to build its own internal implementation via a factory method:

| Connector | Factory method | Returns |
|-----------|---------------|---------|
| `LocalFilesSource` | `create_watcher(data_source_id)` | `LocalFileWatcher` |
| `PgVectorTarget` | `create_sync_target()` | `PgVectorSyncTarget` |
| `OpenAIEmbedder` | `create_provider()` | `OpenAIProvider` |

`app.py` now calls these methods generically, with source registration folded into the same loop:

```python
# app.py (now) — no specific provider names anywhere
sync_engine = SyncEngine(manifest=manifest, sync_target=config.target.create_sync_target())
embedder = Embedder(provider=config.embeddings.create_provider())

active_source_ids = []
source_registry = {}
for source in config.sources:
    data_source_id = manifest.register_source(source.source_type, source.source_identifier)
    active_source_ids.append(data_source_id)
    source_registry[data_source_id] = source
    monitor.add_watcher(source.create_watcher(data_source_id))
manifest.delete_stale_sources(active_source_ids)
```

Adding a `PineconeTarget` in the future means writing `PineconeTarget.create_sync_target()` — `app.py` never needs to change.

### Why `SyncEngine` Also Had to Change

`SyncEngine` previously held a `pg_connstr: str` and called `from . import pgvector` directly at module level. This meant `SyncEngine` was permanently coupled to Postgres, regardless of what target the user configured.

The fix: `SyncEngine` now accepts a `SyncTarget` ABC instance. It calls `sync_target.ensure_table_exists(config)` and `sync_target.sync_batch(upserts, deletes, config)` — both defined on the abstract base. The pgvector logic lives entirely inside `PgVectorSyncTarget`, which `SyncEngine` never imports or references.

### Why the ABCs in `connectors/base.py`

Without the abstract base classes (`SourceConnector`, `TargetConnector`, `EmbedderConfig`), `defineConfig` would have had to validate by checking `isinstance(source, LocalFilesSource)` — hardcoding the only allowed type. With the ABCs, `defineConfig` checks `isinstance(source, SourceConnector)` — any future connector that inherits from `SourceConnector` is accepted automatically, with no changes to `defineConfig`.

`SourceConnector` declares six abstract members:
- `source_type: str` — a stable string identifying the connector category (e.g. `"local"`)
- `source_identifier: str` — a stable string identifying the specific connection (e.g. the normalized absolute path)
- `create_watcher(data_source_id: int)` — returns a `BaseWatcher` bound to this source
- `async fetch(task: SyncTask) -> SpruceFile` — reads the file from this source and returns a `SpruceFile`
- `display_name(identifier: str) -> str` — converts a task identifier into a short human-readable label for log output
- `decode_content(raw_content: bytes) -> str` — converts the raw bytes fetched from a source into the string passed to `@transform` as `file_props["raw_content"]`

`display_name` exists because `Coordinator` needs to log a short label for each file it processes, but what makes a good label is connector-specific. For local files, `pathlib.Path(identifier).name` (the filename) is natural. For an S3 connector, the object key without the bucket prefix makes more sense. For a database connector, a primary key or a truncated row description might be right. Without this method, `Coordinator` would have to assume identifiers are POSIX paths — which is fine today but silently wrong for any non-filesystem source.

`decode_content` exists for the same reason applied to encoding. `Coordinator` needs to convert `SpruceFile.raw_content` (always `bytes`) into the string that `@transform` receives. Previously it called `.decode(errors="replace")` directly, which silently assumes UTF-8. A connector working with Latin-1 documents, a non-text API response, or any other encoding would produce garbled output with no error. The connector knows the right decoding — `LocalFilesSource` uses UTF-8 with error replacement; a future connector can override to use a different codec or raise explicitly for binary content that should never be decoded as text.

These six members drive the source registration, watcher scoping, routing, logging, and content-decoding systems described above.

### Why `data_source_id` Had to Flow Into `SyncTask`

Before these changes, `SyncTask` had a `source_type` field — a string like `"local"` — and `Coordinator` had a `FetcherRegistry` that matched on it:

```python
# FetcherRegistry (removed)
match task.source_type:
    case "local":
        return LocalFileFetcher(task, data_source_id)
```

With multiple sources of the same type (e.g. two `LocalFilesSource` connectors), `source_type` alone can't identify which connector produced a task — both have `source_type="local"`. The `data_source_id` integer is the actual discriminator.

`SyncTask` now carries `data_source_id: int`, set by the watcher when it constructs the task. `Coordinator` receives a `source_registry: dict[int, SourceConnector]` and dispatches directly:

```python
source = self._source_registry[task.data_source_id]
spruce_file = await source.fetch(task)
```

This eliminates `LocalFileFetcher` and `FetcherRegistry` entirely. The fetch logic (open file, stat, hash) moved to `LocalFilesSource.fetch()`, where it belongs — the connector knows how to read its own files.

### Why `source_type` Is Now a Property on `BaseWatcher`

`LocalFileWatcher` constructs `SyncTask` objects and needs to set the `source_type` field on them. That string should come from the connector (which owns it), not be hardcoded in the watcher class. `LocalFilesSource.create_watcher(data_source_id)` therefore passes `self.source_type` to the watcher at construction.

`BaseWatcher` declares `source_type` as an abstract property, enforcing this contract on all future watcher implementations.

### Why Config Is the Source of Truth for Stale Detection

An alternative design would delete stale `data_sources` rows only after confirming that a source has become permanently unavailable (e.g., after a connection timeout). This would be wrong: if a network mount is temporarily down, SpruceUp would delete all that source's file and chunk records from the manifest. When the mount came back, everything would need to be fully re-indexed.

The correct rule: a source is stale if and only if it no longer appears in `config.sources`. If a source temporarily fails to connect, `_with_retry` in `monitor.py` keeps retrying with exponential backoff — it never triggers deletion. The user's config is the authoritative record of which sources should exist.

### Watcher Retry Logging

`_with_retry` previously caught all exceptions from watchers and slept silently. A misconfigured `watched_dir` (or a temporarily unavailable mount point) would spin forever in the background with no console output. Now each failure logs a warning:

```
WARNING  Watcher failed (attempt 3) — retrying in 4s: [Errno 2] No such file or directory: '/mnt/nas/corpus'
```

The retry behavior itself — infinite retries with exponential backoff, capped at 60 seconds — is intentional and unchanged. Only the silence was a bug.

---

## File Inventory

### New Files

| File | Purpose |
|------|---------|
| `spruceup/config.py` | `SpruceUpConfig` dataclass + `defineConfig()` function; validates connector types against ABCs |
| `spruceup/connectors/__init__.py` | Re-exports all user-facing connector classes |
| `spruceup/connectors/base.py` | Abstract base classes: `SourceConnector` (with `source_type`, `source_identifier`, `create_watcher`, `fetch`, `display_name`, `decode_content`), `TargetConnector`, `EmbedderConfig`, `SyncTarget` |
| `spruceup/connectors/sources/__init__.py` | Re-exports `LocalFilesSource` |
| `spruceup/connectors/sources/local.py` | `LocalFilesSource(watched_dir)` — implements all `SourceConnector` abstract members including `fetch()`, `display_name()`, and `decode_content()` |
| `spruceup/connectors/targets/__init__.py` | Re-exports `PgVectorTarget` |
| `spruceup/connectors/targets/pgvector.py` | `PgVectorTarget(connstr, table, schema, primary_key)` — implements `create_sync_target()` returning `PgVectorSyncTarget`; also contains `PgVectorSyncTarget` and its private helpers (`_ensure_table_exists`, `_upsert_chunks`, `_delete_chunks`) |
| `spruceup/connectors/embedders/__init__.py` | Re-exports `OpenAIEmbedder` |
| `spruceup/connectors/embedders/openai.py` | `OpenAIEmbedder(model)` — implements `create_provider()` returning `OpenAIProvider` |

### Modified Files

| File | What Changed |
|------|-------------|
| `spruceup/__init__.py` | Added exports: `defineConfig`, `LocalFilesSource`, `PgVectorTarget`, `OpenAIEmbedder` |
| `spruceup/pipeline_validator.py` | Replaced 5-constant loop with a single check: `isinstance(pipeline.config, SpruceUpConfig)` |
| `spruceup/app.py` | Connector-agnostic; source registration loop builds `source_registry` and `active_source_ids`; calls `delete_stale_sources` at startup; passes `source_registry` to `Coordinator` |
| `spruceup/sync_engine/sync_engine.py` | Accepts `sync_target: SyncTarget` instead of `pg_connstr: str`; calls `sync_target` methods instead of importing pgvector directly |
| `spruceup/coordinator.py` | Removed `LocalFileFetcher` and `FetcherRegistry`; accepts `source_registry: dict[int, SourceConnector]`; dispatches `source_registry[task.data_source_id].fetch(task)`; log labels via `source.display_name(task.identifier)`; content decoding via `source.decode_content(raw_content)` |
| `spruceup/monitoring/monitor.py` | `BaseWatcher` gains abstract `source_type` property; `LocalFileWatcher` takes `source_type` + `data_source_id` params; `_catch_up` and `_watch` filter manifest queries by `data_source_id`; all `SyncTask` constructions use `self._source_type`; `_with_retry` logs a warning on each failure |
| `spruceup/monitoring/tasks.py` | `SyncTask` gains `data_source_id: int` field (default `0`) |
| `spruceup/manifest.py` | Added `register_source(source_type, source_identifier) -> int` and `delete_stale_sources(active_ids)` |
| `spruceup/db.py` | `data_sources` gains `source_identifier TEXT NOT NULL` + `UNIQUE(source_type, source_identifier)`; hardcoded startup row removed |
| `spruceup_pipeline.py` | Replaced 5 loose constants with `defineConfig(sources=[...], target=..., embeddings=...)` |
| `CLAUDE.md` | Updated file map, pipeline customisation docs, and key invariants |
| `tests/test_sync_engine.py` | Fixture calls `manifest.register_source()` to satisfy FK constraint (no more hardcoded row); `MockSyncTarget(SyncTarget)` replaces `MockPgConn` + `psycopg.connect` patch |

### Deleted (Implicitly Replaced)

| What | Replaced By |
|------|-------------|
| `spruceup/sync_engine/target_db.py` | `connectors/targets/pgvector.py` — same Postgres logic, now as `PgVectorSyncTarget` class |
| `LocalFileFetcher` class in `coordinator.py` | `LocalFilesSource.fetch()` — fetch logic lives on the connector |
| `FetcherRegistry` class in `coordinator.py` | `source_registry: dict[int, SourceConnector]` — routing by `data_source_id`, not source type string |

---

## Current App Flow

```
┌─────────────────────────────────────────────────────────┐
│                  spruceup_pipeline.py                   │
│                                                         │
│  @dataclass class LectureChunk: ...                     │
│                                                         │
│  @transform                                             │
│  async def build_chunks(*, file_props, embed): ...      │
│    └─ registers with TransformTracker singleton         │
│                                                         │
│  config = defineConfig(                                 │
│    sources=[LocalFilesSource(watched_dir=...)],         │
│    target=PgVectorTarget(connstr=..., table=...),       │
│    embeddings=OpenAIEmbedder(model=...),                │
│  )  ← validates all fields eagerly at import time       │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                       cli.py                            │
│                                                         │
│  importlib.import_module("spruceup_pipeline")           │
│  validate_pipeline(pipeline)                            │
│    ├─ pipeline.config is SpruceUpConfig? ✓              │
│    └─ @transform registered? ✓                          │
│  asyncio.run(app.run(pipeline))                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                       app.py                            │
│                 (connector-agnostic)                    │
│                                                         │
│  SyncEngine(sync_target=config.target                   │
│               .create_sync_target())    ──────────────► PgVectorSyncTarget
│  Embedder(provider=config.embeddings                    │
│               .create_provider())       ──────────────► OpenAIProvider
│                                                         │
│  for source in config.sources:                          │
│    id = manifest.register_source(                       │
│           source.source_type,                           │
│           source.source_identifier)                     │
│    source_registry[id] = source                         │
│    monitor.add_watcher(source.create_watcher(id)) ────► LocalFileWatcher(id)
│  manifest.delete_stale_sources(active_source_ids)       │
│                                                         │
│  Coordinator(..., source_registry=source_registry)      │
│  asyncio.gather(monitor.run(), coordinator.run())       │
└──────────────────┬──────────────────┬───────────────────┘
                   │                  │
        ┌──────────┘                  └──────────┐
        ▼                                        ▼
┌───────────────────┐              ┌─────────────────────────┐
│     Monitor       │              │       Coordinator        │
│                   │   SyncTask   │                          │
│ LocalFileWatcher  │   (carries   │ source_registry          │
│  ._catch_up()     │──data_source─│  [task.data_source_id]  │
│  ._watch()        │   _id) ────► │  .fetch(task)            │
│                   │   queue      │         │                │
│ _BufferedQueue    │              │         ▼                │
│  holds events     │              │  @transform(file_props,  │
│  during catch_up  │              │            embed)        │
└───────────────────┘              │         │                │
                                   │         ▼                │
                                   │  Embedder.process_chunks │
                                   │  → OpenAIProvider        │
                                   │    .embed_batch()        │
                                   │         │                │
                                   │         ▼                │
                                   │  validate_schema_objects │
                                   │         │                │
                                   └─────────┼────────────────┘
                                             │
                                             ▼
                          ┌──────────────────────────────────┐
                          │          SyncEngine              │
                          │                                  │
                          │  reconcile(files):               │
                          │    diff new vs manifest chunks   │
                          │    SyncTarget.sync_batch(        │
                          │      upserts, deletes, config    │
                          │    ) ──────────────────────────► PgVectorSyncTarget
                          │    Manifest: upsert/delete       │   └─ psycopg transaction
                          │             chunks + file row    │      INSERT + DELETE
                          │                                  │
                          │  move_file(old, new):            │
                          │    Manifest only — no SyncTarget │
                          │    (vectors are content-keyed)   │
                          │                                  │
                          │  delete_file(path):              │
                          │    SyncTarget.sync_batch(        │
                          │      [], chunk_pks, config       │
                          │    )                             │
                          │    Manifest: delete chunks       │
                          │             + file row           │
                          └──────────────────────────────────┘


SQLite (spruceup_manifest.db)                       Postgres
──────────────────────────────────────────          ────────────────────────
data_sources (id, source_type,                      <user-defined table>
              source_identifier)                    (schema from LectureChunk
files (id, file_path, inode,                         dataclass, auto-created)
       content_hash, mtime,
       data_source_id FK, file_type)
chunks (id, file_id FK,
        user_chunk_object_hash,
        user_chunk_object)
transform_hashes
```

### Adding a New Connector (Future Reference)

To add a new source (e.g., `S3Source`):
1. Create `spruceup/connectors/sources/s3.py` — `@dataclass class S3Source(SourceConnector)`:
   - `source_type` property returns `"s3"` (a stable, unique string for this connector type)
   - `source_identifier` property returns a stable string for the specific bucket/prefix — used for idempotent DB registration and stale detection across restarts
   - `create_watcher(data_source_id)` returns `S3Watcher(bucket, prefix, data_source_id, self.source_type)`
   - `async fetch(task) -> SpruceFile` downloads the object from S3 and returns a populated `SpruceFile`
   - `display_name(identifier: str) -> str` returns a short readable label for logging (e.g. the object key without the bucket prefix)
   - `decode_content(raw_content: bytes) -> str` decodes the fetched bytes into a string for `@transform` (e.g. UTF-8 for text objects; raise explicitly for binary-only buckets)
2. Create `spruceup/monitoring/s3_watcher.py` — `S3Watcher(BaseWatcher)` implementing `source_type`, `_catch_up`, and `_watch`, filtering all manifest queries by `self._data_source_id`
3. Export from `spruceup/connectors/sources/__init__.py` and `spruceup/connectors/__init__.py`
4. `app.py` needs zero changes — source registration, watcher creation, routing, and stale detection all generalize automatically

Same pattern applies for new `TargetConnector` or `EmbedderConfig` implementations.
