## SpruceUp

**SpruceUp** is a standalone system for making automated, incremental updates to a vector database.

## Installation

Add SpruceUp to your project (e.g., with `poetry`, `pip`, or `uv`):

```bash
poetry add spruceup-ai # OR
pip install spruceup-ai # OR
uv add spruceup-ai
```

---

## Setup

Create a file named `spruceup_pipeline.py` in your project directory. This is the user-authored entry point SpruceUp loads at startup. It must export a single `config` variable built with `define_config()`.

```python
# spruceup_pipeline.py
import re
import os
from dataclasses import dataclass
from spruceup import define_config, FileProps, LocalFilesSource, PgVectorTarget, OpenAIEmbedder

@dataclass
class ArticleChunk:
    title: str
    content: str
    embedding: list[float]

def split_into_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

async def transform(*, file_props: FileProps, embed) -> list[ArticleChunk]:
    paragraphs = split_into_paragraphs(file_props.raw_content)
    embeddings = await embed(paragraphs)
    return [
        ArticleChunk(title=file_props.display_name, content=para, embedding=emb)
        for para, emb in zip(paragraphs, embeddings)
    ]

config = define_config(
    sources=[LocalFilesSource(watched_dir="./articles")],
    target=PgVectorTarget(
        connstr=os.environ["PG_CONNSTR"],
        table="article_chunks",
        schema=ArticleChunk,
        vector_column="embedding",
    ),
    embedder=OpenAIEmbedder(api_key=os.environ["OPENAI_API_KEY"]),
    transform=transform,
)
```

---

## Running SpruceUp

From the directory containing your `spruceup_pipeline.py` file, with your virtual environment activated:

```bash
spruceup start
```

SpruceUp will scan your sources, sync any files not yet in the manifest, then enter a watch loop for incremental updates.

---

## Imports

Everything you need is importable from the top-level `spruceup` package:

```python
from spruceup import (
    define_config,
    FileProps,

    # Sources
    LocalFilesSource,
    GoogleDriveSource,

    # Targets
    PgVectorTarget,
    PineconeTarget,
    WeaviateTarget,

    # Embedders
    OpenAIEmbedder,
    CohereEmbedder,
    GeminiEmbedder,
    VoyageAIEmbedder,

    # Utilities
    memoize,
)
```

---

## `define_config()`

```python
config = define_config(
    sources=[...],
    target=...,
    embedder=...,
    transform=...,
)
```

All parameters are keyword-only.

| Parameter     | Type                    | Required | Default | Description                           |
| ------------- | ----------------------- | -------- | ------- | ------------------------------------- |
| `sources`     | `list[SourceConnector]` | Yes      | —       | At least one source connector         |
| `target`      | `TargetConnector`       | Yes      | —       | Where synced chunks are written       |
| `embedder`    | `EmbedderConnector`     | Yes      | —       | Generates embeddings for your chunks  |
| `transform`   | `async callable`        | Yes      | —       | Converts a file into a list of chunks |
| `cache_files` | `bool`                  | No       | `False` | Cache raw file bytes in the manifest  |

---

## The Transform Function

The transform function is where you split, enrich, and embed your documents. SpruceUp calls it for every file that changes. This function **must** be async.

```python
async def transform(*, file_props: FileProps, embed) -> list[YourSchema]:
    ...
```

### `FileProps`

| Field          | Type           | Description                                                  |
| -------------- | -------------- | ------------------------------------------------------------ |
| `raw_content`  | `str \| bytes` | File content. Text formats are decoded as UTF-8; binary formats like PDF are passed through as raw `bytes`. |
| `display_name` | `str`          | The filename                                                 |
| `file_type`    | `str`          | File extension (e.g. `"txt"`, `"pdf"`)                       |

### `embed`

`embed` is an async callable that takes a list of strings and returns a list of embedding vectors:

```python
embeddings: list[list[float]] = await embed(["chunk one", "chunk two"])
```

### Chunk Schema

Your transform returns a list of instances of a user-defined dataclass. SpruceUp uses this schema for diffing and for writing to the target store. Define it as a plain dataclass:

```python
@dataclass
class MyChunk:
    title: str
    text: str
    embedding: list[float]
```

All target connectors support `str`, `int`, `float`, `bool`, and `list[float]` as field types. Use `list[float]` for your embedding vector. You do not need to define an `id` field. SpruceUp generates one from each chunk's content hash.

---

## Source Connectors

### `LocalFilesSource`

Watches a local directory for file changes.

```python
LocalFilesSource(watched_dir="./data")
```

| Parameter     | Type  | Required | Default | Description                    |
| ------------- | ----- | -------- | ------- | ------------------------------ |
| `watched_dir` | `str` | Yes      | —       | Path to the directory to watch |

---

### `GoogleDriveSource`

Watches a Google Drive folder for file changes. Requires the `drive.readonly` OAuth scope.

```python
GoogleDriveSource(
    watched_dir="<folder-id>",
    on_token_expired=get_access_token,
    recursive=True,
)
```

| Parameter          | Type                | Required | Default | Description                                                  |
| ------------------ | ------------------- | -------- | ------- | ------------------------------------------------------------ |
| `watched_dir`      | `str`               | Yes      | —       | Google Drive folder ID                                       |
| `on_token_expired` | `Callable[[], str]` | Yes      | —       | Called when the access token expires; must return a fresh token string |
| `recursive`        | `bool`              | No       | `True`  | Whether to watch subfolders                                  |

The `on_token_expired` callback is invoked whenever the connector needs a new OAuth token. It should return a valid access token or raise an exception.

---

## Target Connectors

### `PgVectorTarget`

Syncs chunks to a PostgreSQL table using the `pgvector` extension.

```python
PgVectorTarget(
    connstr="postgresql://user:pass@localhost/mydb",
    table="my_chunks",
    schema=MyChunk,
    vector_column="embedding",
)
```

| Parameter       | Type   | Required | Default | Description                                    |
| --------------- | ------ | -------- | ------- | ---------------------------------------------- |
| `connstr`       | `str`  | Yes      | —       | PostgreSQL connection string                   |
| `table`         | `str`  | Yes      | —       | Table name                                     |
| `schema`        | `type` | Yes      | —       | Your chunk dataclass                           |
| `vector_column` | `str`  | Yes      | —       | Field name on your schema that holds the vector |

SpruceUp creates the table and its columns automatically based on your schema's type hints. The `pgvector` extension must be installed on your database.

---

### `PineconeTarget`

Syncs chunks to a Pinecone index.

```python
PineconeTarget(
    api_key="pc-...",
    index_name="my-index",
    schema=MyChunk,
    vector_column="embedding",
    namespace="",
    metric="cosine",
    cloud="aws",
    region="us-east-1",
)
```

| Parameter       | Type          | Required | Default       | Description                                                 |
| --------------- | ------------- | -------- | ------------- | ----------------------------------------------------------- |
| `api_key`       | `str \| None` | Yes      | —             | Pinecone API key                                            |
| `index_name`    | `str`         | Yes      | —             | Name of the Pinecone index                                  |
| `schema`        | `type`        | Yes      | —             | Your chunk dataclass                                        |
| `vector_column` | `str`         | Yes      | —             | Field name on your schema that holds the vector             |
| `namespace`     | `str`         | No       | `""`          | Namespace within the index                                  |
| `metric`        | `str`         | No       | `"cosine"`    | Distance metric (`"cosine"`, `"euclidean"`, `"dotproduct"`) |
| `cloud`         | `str`         | No       | `"aws"`       | Cloud provider                                              |
| `region`        | `str`         | No       | `"us-east-1"` | Cloud region                                                |

---

### `WeaviateTarget`

Syncs chunks to a Weaviate collection.

```python
# Local instance
WeaviateTarget(
    collection_name="MyChunks",
    schema=MyChunk,
    vector_column="embedding",
    url="http://localhost:8080",
)

# Weaviate Cloud
WeaviateTarget(
    collection_name="MyChunks",
    schema=MyChunk,
    vector_column="embedding",
    cluster_url="https://my-cluster.weaviate.network",
    api_key="wvp-...",
)
```

| Parameter         | Type          | Required | Default                   | Description                               |
| ----------------- | ------------- | -------- | ------------------------- | ----------------------------------------- |
| `collection_name` | `str`         | Yes      | —                         | Weaviate collection name                  |
| `schema`          | `type`        | Yes      | —                         | Your chunk dataclass                      |
| `vector_column`   | `str`         | Yes      | —                         | Field name on your schema that holds the vector |
| `url`             | `str`         | No       | `"http://localhost:8080"` | URL for a local Weaviate instance         |
| `cluster_url`     | `str \| None` | No       | `None`                    | URL for a Weaviate Cloud cluster          |
| `api_key`         | `str \| None` | No       | `None`                    | API key for Weaviate Cloud authentication |

Use either `url` for a local instance or `cluster_url` + `api_key` for a cloud deployment.

---

## Embedder Connectors

SpruceUp runs a health check at startup that embeds a test string and reads the actual output size from the API. The `embedding_dimensions` parameter is optional on all embedders. If omitted, the dimension is detected automatically. If provided, SpruceUp validates it matches what the API actually returns and raises an error if not.

### `OpenAIEmbedder`

```python
OpenAIEmbedder(
    api_key="sk-...",
    model="text-embedding-3-small",
    max_batch_size=150,
    embedding_dimensions=None,
)
```

| Parameter              | Type                        | Required | Default                    | Description                |
| ---------------------- | --------------------------- | -------- | -------------------------- | -------------------------- |
| `api_key`              | `str \| Callable[[], str]`  | Yes      | —                          | OpenAI API key, or a callable that returns one |
| `model`                | `str`                       | No       | `"text-embedding-3-small"` | Embedding model            |
| `max_batch_size`       | `int`                       | No       | `150`                      | Max texts per API call     |
| `embedding_dimensions` | `int \| None`               | No       | `None`                     | Override output dimensions. If omitted, SpruceUp reads the actual dimension from the API at startup. |

---

### `CohereEmbedder`

```python
CohereEmbedder(
    api_key="...",
    model="embed-v4.0",
    max_batch_size=96,
    embedding_dimensions=None,
)
```

| Parameter              | Type                        | Required | Default        | Description                |
| ---------------------- | --------------------------- | -------- | -------------- | -------------------------- |
| `api_key`              | `str \| Callable[[], str]`  | Yes      | —              | Cohere API key, or a callable that returns one |
| `model`                | `str`                       | No       | `"embed-v4.0"` | Embedding model            |
| `max_batch_size`       | `int`                       | No       | `96`           | Max texts per API call     |
| `embedding_dimensions` | `int \| None`               | No       | `None`         | Override output dimensions. If omitted, SpruceUp reads the actual dimension from the API at startup. |

When using an `embed-v4` model with a custom `embedding_dimensions`, the value must be one of `256`, `512`, `1024`, or `1536`.

---

### `GeminiEmbedder`

```python
GeminiEmbedder(
    api_key="...",
    model="gemini-embedding-001",
    max_batch_size=100,
)
```

| Parameter              | Type                        | Required | Default                  | Description                              |
| ---------------------- | --------------------------- | -------- | ------------------------ | ---------------------------------------- |
| `api_key`              | `str \| Callable[[], str]`  | Yes      | —                        | Google Generative AI API key, or a callable that returns one |
| `model`                | `str`                       | No       | `"gemini-embedding-001"` | Embedding model                          |
| `max_batch_size`       | `int`                       | No       | `100`                    | Max texts per API call (hard limit: 100) |
| `embedding_dimensions` | `int \| None`               | No       | `None`                   | Override output dimensions. If omitted, SpruceUp reads the actual dimension from the API at startup. |

---

### `VoyageAIEmbedder`

```python
VoyageAIEmbedder(
    api_key="...",
    model="voyage-4-large",
    max_batch_size=150,
    embedding_dimensions=None,
)
```

| Parameter              | Type                        | Required | Default            | Description                |
| ---------------------- | --------------------------- | -------- | ------------------ | -------------------------- |
| `api_key`              | `str \| Callable[[], str]`  | Yes      | —                  | Voyage AI API key, or a callable that returns one |
| `model`                | `str`                       | No       | `"voyage-4-large"` | Embedding model            |
| `max_batch_size`       | `int`                       | No       | `150`              | Max texts per API call     |
| `embedding_dimensions` | `int \| None`               | No       | `None`             | Override output dimensions. If omitted, SpruceUp reads the actual dimension from the API at startup. |

When using a `voyage-4` model with a custom `embedding_dimensions`, the value must be one of `256`, `512`, `1024`, or `2048`.

---

## `@memoize`

The `memoize` decorator caches the results of expensive subfunctions inside your transform. Results are stored in the SpruceUp manifest (a local SQLite database), scoped per file and invalidated automatically when the decorated function's body changes.

```python
from spruceup import memoize
import asyncio

@memoize(return_type=str)
async def summarize(text: str) -> str:
    # expensive LLM call
    ...

async def transform(*, file_props: FileProps, embed) -> list[MyChunk]:
    chunk_strs = split_into_chunks(file_props.raw_content)
    # summarize each chunk concurrently; results are cached per file
    summaries = await asyncio.gather(*[summarize(c) for c in chunk_strs])
    embeddings = await embed(chunk_strs)
    return [
        MyChunk(content=c, summary=s, embedding=e)
        for c, s, e in zip(chunk_strs, summaries, embeddings)
    ]
```

| Parameter     | Type   | Required | Description                                                  |
| ------------- | ------ | -------- | ------------------------------------------------------------ |
| `return_type` | `type` | Yes      | Return type of the decorated function — used for serialization |

Supported return types: `str`, `int`, `float`, `bool`, `list`, `dict`.

`memoize` only works on `async` functions. Decorating a sync function raises a `TypeError`. It can only be used inside a transform function. Calling a memoized function outside of a transform context will raise a `RuntimeError`.