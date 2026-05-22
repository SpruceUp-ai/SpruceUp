# User Pipeline Changes — `defineConfig` Refactor

## What Changed and Why

### The Problem with Loose Constants

Before this change, users configured SpruceUp by defining five module-level constants in their `spruceup_pipeline.py`:

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

---

## Why the Overhaul Was Larger Than "Just Wrap the Constants"

This is the key question. On the surface, `defineConfig` seems like a simple wrapper. Why did it require a new connector architecture?

### The Problem: `app.py` Was the Integration Point

Before the change, `app.py` hardcoded every specific implementation:

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
| `LocalFilesSource` | `create_watcher()` | `LocalFileWatcher` |
| `PgVectorTarget` | `create_sync_target()` | `PgVectorSyncTarget` |
| `OpenAIEmbedder` | `create_provider()` | `OpenAIProvider` |

`app.py` now calls these methods generically:

```python
# app.py (after) — no specific provider names anywhere
sync_engine = SyncEngine(manifest=manifest, sync_target=config.target.create_sync_target())
embedder = Embedder(provider=config.embeddings.create_provider())
for source in config.sources:
    monitor.add_watcher(source.create_watcher())
```

Adding a `PineconeTarget` in the future means writing `PineconeTarget.create_sync_target()` — `app.py` never needs to change.

### Why `SyncEngine` Also Had to Change

`SyncEngine` previously held a `pg_connstr: str` and called `from . import pgvector` directly at module level. This meant `SyncEngine` was permanently coupled to Postgres, regardless of what target the user configured.

The fix: `SyncEngine` now accepts a `SyncTarget` ABC instance. It calls `sync_target.ensure_table_exists(config)` and `sync_target.sync_batch(upserts, deletes, config)` — both defined on the abstract base. The pgvector logic lives entirely inside `PgVectorSyncTarget`, which `SyncEngine` never imports or references.

### Why the ABCs in `connectors/base.py`

Without the abstract base classes (`SourceConnector`, `TargetConnector`, `EmbedderConfig`), `defineConfig` would have had to validate by checking `isinstance(source, LocalFilesSource)` — hardcoding the only allowed type. With the ABCs, `defineConfig` checks `isinstance(source, SourceConnector)` — any future connector that inherits from `SourceConnector` is accepted automatically, with no changes to `defineConfig`.

---

## File Inventory

### New Files

| File | Purpose |
|------|---------|
| `spruceup/config.py` | `SpruceUpConfig` dataclass + `defineConfig()` function; validates connector types against ABCs |
| `spruceup/connectors/__init__.py` | Re-exports all user-facing connector classes |
| `spruceup/connectors/base.py` | Abstract base classes: `SourceConnector`, `TargetConnector`, `EmbedderConfig` — each declares a factory method |
| `spruceup/connectors/sources/__init__.py` | Re-exports `LocalFilesSource` |
| `spruceup/connectors/sources/local.py` | `LocalFilesSource(watched_dir)` — implements `create_watcher()` returning `LocalFileWatcher` |
| `spruceup/connectors/targets/__init__.py` | Re-exports `PgVectorTarget` |
| `spruceup/connectors/targets/pgvector.py` | `PgVectorTarget(connstr, table, schema, primary_key)` — implements `create_sync_target()` returning `PgVectorSyncTarget` |
| `spruceup/connectors/embedders/__init__.py` | Re-exports `OpenAIEmbedder` |
| `spruceup/connectors/embedders/openai.py` | `OpenAIEmbedder(model)` — implements `create_provider()` returning `OpenAIProvider` |
| `spruceup/sync_engine/target_connectors/__init__.py` | Makes `target_connectors` a package |
| `spruceup/sync_engine/target_connectors/base.py` | `SyncTarget` ABC — defines `ensure_table_exists` and `sync_batch` interface |
| `spruceup/sync_engine/target_connectors/pgvector.py` | `PgVectorSyncTarget(connstr)` — implements the Postgres-specific sync logic (upsert, delete, table creation) |

### Modified Files

| File | What Changed |
|------|-------------|
| `spruceup/__init__.py` | Added exports: `defineConfig`, `LocalFilesSource`, `PgVectorTarget`, `OpenAIEmbedder` |
| `spruceup/pipeline_validator.py` | Replaced 5-constant loop with a single check: `isinstance(pipeline.config, SpruceUpConfig)` |
| `spruceup/app.py` | Now connector-agnostic: uses factory methods, removed all hardcoded provider imports |
| `spruceup/sync_engine/sync_engine.py` | Accepts `sync_target: SyncTarget` instead of `pg_connstr: str`; calls `sync_target` methods instead of importing pgvector directly |
| `spruceup_pipeline.py` | Replaced 5 loose constants with `defineConfig(sources=[...], target=..., embeddings=...)` |
| `CLAUDE.md` | Updated file map, pipeline customisation docs, and key invariants |
| `tests/test_sync_engine.py` | Replaced `MockPgConn` + `psycopg.connect` patch with `MockSyncTarget(SyncTarget)` |

### Deleted (implicitly replaced)

The old `spruceup/sync_engine/target_db.py` (Postgres functions as module-level fns) was previously renamed to `target_connectors/pgvector.py` and converted into the `PgVectorSyncTarget` class in this change.

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
│  config = pipeline.config                               │
│                                                         │
│  SyncEngine(                                            │
│    manifest=Manifest(MANIFEST_PATH),                    │
│    sync_target=config.target.create_sync_target()       │──► PgVectorSyncTarget
│  )                                                      │
│  sync_engine.define_target_table(table, schema, pk)     │
│                                                         │
│  Embedder(                                              │
│    provider=config.embeddings.create_provider()         │──► OpenAIProvider
│  )                                                      │
│                                                         │
│  Coordinator(queue, transform_fn, embedder,             │
│              sync_engine, schema, pk)                   │
│                                                         │
│  Monitor(queue, manifest, transform_tracker)            │
│  for source in config.sources:                          │
│    monitor.add_watcher(source.create_watcher())         │──► LocalFileWatcher
│                                                         │
│  asyncio.gather(monitor.run(), coordinator.run())       │
└──────────────────┬──────────────────┬───────────────────┘
                   │                  │
        ┌──────────┘                  └──────────┐
        ▼                                        ▼
┌───────────────────┐              ┌─────────────────────────┐
│     Monitor       │              │       Coordinator        │
│                   │              │                          │
│ LocalFileWatcher  │   SyncTask   │ LocalFileFetcher         │
│  ._catch_up()     │─────queue───►│  reads raw file content  │
│  ._watch()        │              │         │                │
│                   │              │         ▼                │
│ _BufferedQueue    │              │  @transform(file_props,  │
│  holds events     │              │            embed)        │
│  during catch_up  │              │         │                │
└───────────────────┘              │         ▼                │
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


SQLite (spruceup_manifest.db)          Postgres
─────────────────────────────          ────────────────────────
data_sources                           <user-defined table>
files (id, file_path, inode,           (schema from CHUNK_SCHEMA
       content_hash, mtime, ...)        dataclass, auto-created)
chunks (id, file_id,
        user_chunk_object_hash,
        user_chunk_object)
transform_hashes
```

### Adding a New Connector (Future Reference)

To add a new source (e.g., `S3Source`):
1. Create `spruceup/connectors/sources/s3.py` — `@dataclass class S3Source(SourceConnector)` with `create_watcher()` returning an `S3Watcher`
2. Create `spruceup/monitoring/s3_watcher.py` — the watcher implementation
3. Export from `spruceup/connectors/sources/__init__.py` and `spruceup/connectors/__init__.py`
4. `app.py` needs zero changes

Same pattern applies for new `TargetConnector` or `EmbedderConfig` implementations.
