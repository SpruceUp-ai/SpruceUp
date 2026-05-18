# Sync Engine Interface Contract

## Overview
The Sync Engine owns all writes to the Postgres vector table and most (all?) writes to the SQLite manifest. It receives data from two sources (other than the dbs): the **user's pipeline file** (which configures the engine at startup) and the **Coordination Layer** (which drives sync operations at runtime).

---

## Wire 1 — User Pipeline

At least when it comes to interacting with the SyncEngine, the user's `spruce_up_pipeline.py` is responsible for two things:

### 1a. Define a chunk schema

The user subclasses `UserDefinedChunkSchema` (?? or defines their own?) as a `@dataclass` to declare the columns of their Postgres vector table:

```python
@dataclass
class MyChunkSchema(UserDefinedChunkSchema):
    # Inherited: id (Any), chunk_text (str), chunk_embedding (list[float])
    source_page: int        # any additional columns they need
```

The field type annotations are used to auto-generate `CREATE TABLE`. Supported mappings (for MVP):

| Python type | Postgres type |
|---|---|
| `str` | `TEXT` |
| `int` | `INTEGER` |
| `float` | `DOUBLE PRECISION` |
| `list[float]` | `vector(1536)` |
| `bytes` | `BYTEA` |
| `bool` | `BOOLEAN` |

### 1b. Call `define_target_table`

```python
engine.define_target_table(
    db_name="rag",
    table_name="vectors",
    schema_from_class=MyChunkSchema,
    primary_key="id",          # must be a field name present in schema_from_class
)
```

Must be called before any other engine method can be invoked since all Engine methods touch the target db.

---

## Wire 2 — Coordination Layer

### `reconcile(files: list[File]) -> None`

Called when one or more files have been embedded or re-embedded and are ready to sync.

The `File` object:
```python
@dataclass
class File:
    file_id: bytes        # hash_file_path(file_path)
    file_path: str
    mtime: float          # os.path.getmtime() at time of processing
    content_hash: bytes   # hash of the raw file bytes
    transform_hash: bytes # hash identifying which transform version was applied
    file_type: str        # e.g. "pdf", "md"
    data_source_id: int   # must reference an existing row in manifest's data_sources table
    raw_content: str | bytes
    parsed_content: str | None
    chunk_strs: list[str]
    chunks: list[ChunkWrapper]
```

Each `ChunkWrapper` in `File.chunks`:
```python
@dataclass
class ChunkWrapper:
    user_chunk: UserDefinedChunkSchema  # fully populated instance of the user's subclass
    user_chunk_object_hash: bytes       # hash_object(user_chunk) — used for change detection
    ordinal: int                        # 1-based position of this chunk within the file
    chunk_id: bytes                     # hash_chunk_id(file_path, ordinal)
```

**Caller guarantees:**
- For each file in `files`: `File.chunks` contains **all** chunks for that file. Missing chunks are treated as deletions. Sending a partial list will cause the absent chunks to be deleted from Postgres.
- `user_chunk_object_hash` must be computed with the exported `hash_object()`. Any other hash function breaks change detection.
(Ultimately, the hash_object function won't live in the source code for the Sync Engine. NOR are we committed to using the Blake2B algorithm which is currently in use in the hash_object function.)
- `chunk_id` must be computed with the exported `hash_chunk_id()`. (see previous note) It must be stable across runs for the same file+ordinal.

**What the engine does on success (in order):**
1. Upserts new/changed chunks to Postgres
2. Deletes orphaned chunks from Postgres
3. Updates the manifest `chunks` table
4. Writes/updates the manifest `files` row ← last write; its presence is the "clean sync" signal

---

### `delete_file(file_id: bytes) -> None`
(IS THIS ACTUALLY CALLED BY THE MANIFEST, RATHER THAN THE COORD LAYER?)

Called when a file has been removed from the corpus.

- `file_id` must be computed with the exported `hash_file_path(file_path)`

**What the engine does on success:**
1. Deletes all vectors for the file from Postgres
2. Deletes all `chunks` rows for the file from the manifest
3. Deletes the `files` row for the file from the manifest

---

## Exported hashing utilities

The Coordination Layer must use these when constructing `File` and `ChunkWrapper` objects — using any other hash function will silently break the engine's ability to match records across runs:

| Function | Used for |
|---|---|
| `hash_file_path(file_path: str) -> bytes` | `File.file_id` |
| `hash_chunk_id(file_path: str, ordinal: int) -> bytes` | `ChunkWrapper.chunk_id` |
| `hash_object(obj) -> bytes` | `ChunkWrapper.user_chunk_object_hash` |

---

## What the engine does NOT do

- Does not read files from disk
- Does not parse, chunk, or embed anything
- Does not compute `content_hash` or `transform_hash` — those come from the Coordination Layer
- Does not populate the `data_sources` table — `data_source_id` values on `File` must already exist
- Does not (currently) retry failed DB writes (but we might move that responsibility into the Sync Engine)
