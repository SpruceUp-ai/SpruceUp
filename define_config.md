# Plan: `defineConfig` API

## Context

SpruceUp currently requires users to define five loose module-level constants in `spruceup_pipeline.py` (`CHUNK_SCHEMA`, `TARGET_TABLE`, `PRIMARY_KEY`, `WATCHED_DIR`, `PG_CONNSTR`). This is fragile — missing or misspelled constants only surface at startup, errors are not co-located with their source, and there is no structure to extend when new source/target/embedding providers are added. The goal is to replace these constants with a single `defineConfig()` call that validates eagerly, reads cleanly, and is naturally extensible to future connectors and providers. The embedding model (currently hardcoded to `text-embedding-3-small` in `embedding.py`) will also move into config so users can select it explicitly.

---

## Target UX

```python
# spruceup_pipeline.py
import os
from dataclasses import dataclass

from spruceup import defineConfig, transform
from spruceup import LocalFilesSource, PgVectorTarget, OpenAIEmbedder


@dataclass
class LectureChunk:
    id: str
    chunk_text: str
    chunk_embedding: list[float]
    lecture_title: str


@transform
async def build_lecture_chunks(*, file_props: dict, embed) -> list[LectureChunk]:
    # embed signature unchanged — still just takes list[str]
    ...


config = defineConfig(
    sources=[
        LocalFilesSource(watched_dir="example/data_corpus"),
        # LocalFilesSource(watched_dir="another/local/dir"),
        # GoogleDriveSource(folder_id="...", credentials=os.environ["GDRIVE_CREDS"]),
        # NotionSource(token=os.environ["NOTION_TOKEN"], database_id="..."),
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

---

## New Files to Create

### `spruceup/sources.py`
```python
from dataclasses import dataclass

@dataclass
class LocalFilesSource:
    watched_dir: str
```
Single source connector for now. Future additions (`S3Source`, etc.) go in this same file until the module warrants splitting into a subpackage.

### `spruceup/targets.py`
```python
from dataclasses import dataclass

@dataclass
class PgVectorTarget:
    connstr: str
    table: str
    schema: type
    primary_key: str
```
Future additions: `PineconeTarget`, `QdrantTarget`, etc.

### `spruceup/embedders.py`
```python
from dataclasses import dataclass

@dataclass
class OpenAIEmbedder:
    model: str = "text-embedding-3-small"
```
Intentionally named `embedders.py` (plural) to avoid conflating with the existing internal `embedding.py` (which stays as-is). Future additions: `CohereEmbedder`, `BedrockEmbedder`, etc.

### `spruceup/config.py`
```python
from dataclasses import dataclass
from .sources import LocalFilesSource
from .targets import PgVectorTarget
from .embedders import OpenAIEmbedder

@dataclass
class SpruceUpConfig:
    sources: list  # list[LocalFilesSource | GoogleDriveSource | ...]
    target: PgVectorTarget
    embeddings: OpenAIEmbedder


def defineConfig(*, sources, target, embeddings) -> SpruceUpConfig:
    # validate sources
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list of source connectors")
    for i, source in enumerate(sources):
        if not isinstance(source, LocalFilesSource):
            raise TypeError(f"sources[{i}] must be a source connector, got {type(source).__name__}")
        if not source.watched_dir:
            raise ValueError(f"sources[{i}]: LocalFilesSource.watched_dir must be a non-empty string")

    # validate target
    if not isinstance(target, PgVectorTarget):
        raise TypeError(f"target must be a PgVectorTarget, got {type(target).__name__}")
    for field_name in ("connstr", "table", "primary_key"):
        if not isinstance(getattr(target, field_name), str) or not getattr(target, field_name):
            raise ValueError(f"PgVectorTarget.{field_name} must be a non-empty string")
    if not isinstance(target.schema, type):
        raise TypeError("PgVectorTarget.schema must be a class (dataclass)")

    # validate embeddings
    if not isinstance(embeddings, OpenAIEmbedder):
        raise TypeError(f"embeddings must be an OpenAIEmbedder, got {type(embeddings).__name__}")
    if not embeddings.model:
        raise ValueError("OpenAIEmbedder.model must be a non-empty string")

    return SpruceUpConfig(sources=sources, target=target, embeddings=embeddings)
```

Validation is **eager** — errors surface the moment `defineConfig()` is called (i.e., at module import time), not at startup.

---

## Files to Modify

### `spruceup/__init__.py`
Add exports for all new public symbols:
```python
from .registry import transform
from .config import defineConfig
from .sources import LocalFilesSource
from .targets import PgVectorTarget
from .embedders import OpenAIEmbedder

__all__ = ["transform", "defineConfig", "LocalFilesSource", "PgVectorTarget", "OpenAIEmbedder"]
```
Single import line covers everything the user needs.

### `spruceup/pipeline_validator.py`
Replace the 5-constant loop entirely. New check:
1. Does `pipeline` have a `config` attribute?
2. Is it a `SpruceUpConfig` instance?
3. Is a `@transform` registered?

Since `defineConfig` already validates field-level correctness eagerly, the validator only confirms the contract exists — no re-validation of individual fields.

```python
from spruceup.config import SpruceUpConfig
import spruceup.registry as registry

def validate_pipeline(pipeline) -> None:
    errors: list[str] = []

    cfg = getattr(pipeline, "config", None)
    if cfg is None:
        errors.append("  config is not defined — call config = defineConfig(...)")
    elif not isinstance(cfg, SpruceUpConfig):
        errors.append(f"  config must be the result of defineConfig(), got {type(cfg).__name__!r}")

    if registry.transform_fn is None:
        errors.append("  no @transform function was registered")

    if errors:
        raise SystemExit("spruceup_pipeline.py is misconfigured:\n" + "\n".join(errors))
```

### `spruceup/app.py`
Replace all `pipeline.*` constant reads with `pipeline.config.*` accessor paths. Wire the embedding model from config into `OpenAIProvider`:

| Before | After |
|--------|-------|
| `monitor.add_watcher(LocalFileWatcher(pipeline.WATCHED_DIR))` | `for s in pipeline.config.sources: monitor.add_watcher(LocalFileWatcher(s.watched_dir))` |
| `pipeline.PG_CONNSTR` | `pipeline.config.target.connstr` |
| `pipeline.TARGET_TABLE` | `pipeline.config.target.table` |
| `pipeline.CHUNK_SCHEMA` | `pipeline.config.target.schema` |
| `pipeline.PRIMARY_KEY` | `pipeline.config.target.primary_key` |
| `OpenAIProvider()` (hardcoded) | `OpenAIProvider(model=pipeline.config.embeddings.model)` |

`Monitor.add_watcher()` already accepts multiple watchers — the infrastructure requires no changes to support multiple sources.

### `spruceup_pipeline.py`
Replace the five loose constants with the `defineConfig(...)` call shown in the **Target UX** section above.

### `CLAUDE.md`
- Update the **Customising the pipeline** section to show `defineConfig()` usage instead of raw constants.
- Update the **File map** table to add entries for `sources.py`, `targets.py`, `embedders.py`, `config.py`.
- Remove the constants-based pipeline example; replace with the new `defineConfig` example.

---

## Files That Do NOT Change

| File | Reason |
|------|--------|
| `spruceup/embedding.py` | Stays internal; `OpenAIProvider` still constructed in `app.py`, now reads `model` from config instead of using the hardcoded default |
| `spruceup/sync_engine/` | No interface changes; `SyncEngine` still receives `pg_connstr` and `define_target_table()` args the same way |
| `spruceup/models.py` | `TargetTableConfig` is an internal model, unchanged |
| `spruceup/monitoring/` | `LocalFileWatcher` still receives `dir_path` from `app.py`, which now reads it from config |
| `tests/` | `test_sync_engine.py` and `test_validation.py` test internal components that don't change |

---

## Verification

```bash
# 1. Run the full test suite — should pass unchanged
poetry run pytest

# 2. Run the app with a valid pipeline
PG_CONNSTR="..." OPENAI_API_KEY="..." spruceup start

# 3. Intentionally break the pipeline to confirm eager validation:
#    - Remove `watched_dir` from LocalFilesSource → ValueError at import time
#    - Set schema to a string instead of a class → TypeError at import time
#    - Remove the `config =` line entirely → SystemExit from pipeline_validator
```
