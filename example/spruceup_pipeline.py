import re
from dataclasses import dataclass

from spruceup import (
    # CohereEmbedder,
    FileProps,
    # GeminiEmbedder,
    GoogleDriveSource,
    LocalFilesSource,
    OpenAIEmbedder,
    PgVectorTarget,
    # PineconeTarget,
    # VoyageAIEmbedder,
    # WeaviateTarget,
    define_config,
    # memoize,
)


@dataclass
class MyChunk:
    title: str
    content: str
    embedding: list[float]


# @memoize(return_type=str)
# async def summarize(text: str) -> str:
#     # expensive LLM call
#     ...


def split_into_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


async def transform(*, file_props: FileProps, embed) -> list[MyChunk]:
    paragraphs = []
    if isinstance(file_props.raw_content, str):
        paragraphs = split_into_paragraphs(file_props.raw_content)

    embeddings = await embed(paragraphs)
    return [
        MyChunk(title=file_props.display_name, content=para, embedding=emb)
        for para, emb in zip(paragraphs, embeddings)
    ]


def get_embedding_api_token() -> str: ...


def get_google_drive_access_token() -> str: ...


config = define_config(
    sources=[
        LocalFilesSource(watched_dir="example/data_corpus"),
        GoogleDriveSource(
            watched_dir="<folder-id>",
            on_token_expired=get_google_drive_access_token,
            recursive=True,
        ),
    ],
    target=PgVectorTarget(
        connstr="postgresql://user:pass@localhost/mydb",
        table="my_chunks",
        schema=MyChunk,
        vector_column="embedding",
    ),
    embedder=OpenAIEmbedder(
        api_key=get_embedding_api_token,
        model="text-embedding-3-small",
        max_batch_size=150,
    ),
    transform=transform,
)
