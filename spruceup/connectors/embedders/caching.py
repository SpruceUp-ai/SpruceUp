import logging

from ..base import EmbedderConnector
from ...manifest import Manifest
from ...utils.hashing import hash_chunk_text
from ...memoize.context import _embed_text_hashes_var

log = logging.getLogger(__name__)


class CachingEmbedder(EmbedderConnector):
    """Embedding cache — the OUTERMOST layer of the embedder chain.

        Coordinator
          └─> CachingEmbedder   (this — hit/miss split, reassembly, provenance)
                └─> EmbeddingBatcher  (coalesces MISSES across files into batches)
                      └─> <EmbedderConnector>  (e.g. `OpenAIEmbedder`; the API call)

    Caching is scoped to chunk text hashes => chunk _content_, not chunk object identity.
    Cache hits are filtered out *before* batching => only misses consume batch slots and API budget.
    A hit requires an exact (text_hash, embedding_spec) match, so a metadata-only
    transform edit (same text, different chunk object) is a ~100% hit at zero API cost.

    Text-hashes computed here are appended, in order, to the _embed_text_hashes_var,
    `ContextVar`.
    The coordinator reads these text-hashes after the transform and writes them
    into `chunks.text_hash`, so the cache lookup and the persisted column share
    a single computation and don't diverge.

    v1 contract: the transform makes one embed() call whose results map 1:1, in
    order, to the chunks it returns.
    """

    def __init__(
        self,
        embedder_connector: EmbedderConnector,
        manifest: Manifest,
        embedding_spec: str,
    ) -> None:
        self._embedder_connector = embedder_connector
        self._manifest = manifest
        self._embedding_spec = embedding_spec
        # Mirror inner's dimensions so app.py's ensure_table_exists path is
        # unaffected if it ever reads through this wrapper.
        self.embedding_dimensions = getattr(embedder_connector, "embedding_dimensions", None)

    @property
    def embedding_spec(self) -> str:
        return self._embedding_spec

    async def embed_batch(self, batch: list[str]) -> list[list[float]]:
        # Pass-through: batching/caching happen at the process_chunks boundary,
        # which is what the transform calls. embed_batch stays a thin delegate so
        # the layer still satisfies EmbedderConnector.
        return await self._embedder_connector.embed_batch(batch)

    async def process_chunks(self, chunks: list[str]) -> list[list[float]]:
        if not chunks:
            return []

        text_hashes = [hash_chunk_text(text) for text in chunks]
        self._record_text_hashes(text_hashes)

        cached = self._manifest.get_embeddings(text_hashes, self._embedding_spec)
        miss_ix = [i for i, h in enumerate(text_hashes) if h not in cached]

        fresh_embeddings: list[list[float]] = []
        if miss_ix:
            fresh_embeddings = await self._embedder_connector.process_chunks([chunks[i] for i in miss_ix])
            self._manifest.set_embeddings(
                [
                    (text_hashes[i], self._embedding_spec, fresh_embeddings[j])
                    for j, i in enumerate(miss_ix)
                ]
            )

        hits = len(chunks) - len(miss_ix)
        pct = (hits / len(chunks) * 100) if chunks else 0
        log.info(
            "[cache] %d hit(s) / %d lookup(s) (%.0f%%, %d API embed(s) saved)",
            hits, len(chunks), pct, hits,
        )

        # Reassemble in ORIGINAL order: hits from cache, misses from fresh.
        # Order is load-bearing — the transform zips the result with chunk text,
        # so we must return the embeddings in the same order as the chunks.
        fresh_embeddings_by_ix = {i: fresh_embeddings[j] for j, i in enumerate(miss_ix)}
        return [
            cached[h] if i not in fresh_embeddings_by_ix else fresh_embeddings_by_ix[i]
            for i, h in enumerate(text_hashes)
        ]

    def _record_text_hashes(self, text_hashes: list[bytes]) -> None:
        # Append (not overwrite) so a transform that makes multiple embed() calls
        # still accumulates all hashes in call order. None when no transform
        # context is active (e.g. a direct embed outside the coordinator).
        bucket = _embed_text_hashes_var.get()
        if bucket is not None:
            bucket.extend(text_hashes)
