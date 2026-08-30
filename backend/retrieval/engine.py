"""Retrieval engine: owns the embedding model, the vector store and BM25.

Deliberately a thin, readable coordinator rather than a framework. Each stage
of the pipeline is a plain method you can read end to end, because the point of
this project is to understand the retrieval path, not to hide it behind an
abstraction that returns answers.
"""

from __future__ import annotations

import logging
import time

from backend.config import Config
from backend.db import Catalogue
from backend.ingestion.chunking import Chunk
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.embeddings import create_embedding_provider
from backend.retrieval.fusion import FusedChunk, reciprocal_rank_fusion
from backend.retrieval.reranker import CrossEncoderReranker
from backend.retrieval.vector_store import ScoredChunk, VectorStore

logger = logging.getLogger(__name__)


class RetrievalEngine:
    def __init__(self, config: Config, catalogue: Catalogue) -> None:
        self.config = config
        self.catalogue = catalogue
        self.embeddings = create_embedding_provider(
            provider=config.embeddings.provider,
            model=config.embeddings.model,
            cache_dir=str(config.storage.model_cache_dir),
            batch_size=config.embeddings.batch_size,
        )
        # Reading the dimension comes from fastembed's model registry, so this
        # does not download or load the model itself -- startup stays fast and
        # the first real download happens when a document is actually indexed.
        self.vector_store = VectorStore(
            path=config.storage.vector_dir, dimension=self.embeddings.dimension
        )
        self.bm25 = BM25Index()
        self.reranker = (
            CrossEncoderReranker(
                model_name=config.reranker.model,
                cache_dir=str(config.storage.model_cache_dir),
                batch_size=config.reranker.batch_size,
            )
            if config.reranker.enabled
            else None
        )

    # --- indexing -----------------------------------------------------------

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Embed chunks and write them to the vector store."""
        if not chunks:
            return
        started = time.perf_counter()
        # search_text, not text: the section heading is part of what gets
        # embedded so a chunk carries the context it was lifted out of.
        vectors = self.embeddings.embed_documents([c.search_text for c in chunks])
        self.vector_store.upsert_chunks(chunks, vectors)
        logger.info(
            "Embedded %d chunks in %.1fs", len(chunks), time.perf_counter() - started
        )

    def rebuild_bm25(self) -> None:
        """Rebuild the lexical index from SQLite, the durable copy."""
        self.bm25.build(self.catalogue.get_chunks())

    def remove_document(self, document_id: str) -> None:
        self.vector_store.delete_document(document_id)
        self.rebuild_bm25()

    # --- retrieval ----------------------------------------------------------

    def dense_search(
        self, query: str, top_k: int | None = None, document_ids: list[str] | None = None
    ) -> list[ScoredChunk]:
        """Semantic search. Finds paraphrases; weak on rare literal terms."""
        top_k = top_k or self.config.retrieval.dense_top_k
        vector = self.embeddings.embed_query(query)
        return self.vector_store.search(vector, top_k=top_k, document_ids=document_ids)

    def bm25_search(
        self, query: str, top_k: int | None = None, document_ids: list[str] | None = None
    ) -> list[ScoredChunk]:
        """Lexical search. Finds exact terminology; blind to paraphrase."""
        top_k = top_k or self.config.retrieval.bm25_top_k
        return self.bm25.search(query, top_k=top_k, document_ids=document_ids)

    def hybrid_search(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        *,
        use_reranker: bool | None = None,
    ) -> list[ScoredChunk]:
        """The full pipeline: retrieve broadly, fuse, then rank precisely.

            query
              |-- dense (top 20) ---.
              |-- bm25  (top 20) ---+--> RRF --> top 24 --> cross-encoder --> top 6

        Both retrievers run over the same filtered document set, so restricting
        the question to three selected papers narrows every stage.
        """
        retrieval = self.config.retrieval
        final_k = top_k or retrieval.final_top_k

        dense = self.dense_search(query, retrieval.dense_top_k, document_ids)
        lexical = self.bm25_search(query, retrieval.bm25_top_k, document_ids)

        candidate_count = max(final_k, self.config.reranker.candidates)
        fused: list[FusedChunk] = reciprocal_rank_fusion(
            dense, lexical,
            k=retrieval.rrf_k,
            dense_weight=retrieval.dense_weight,
            bm25_weight=retrieval.bm25_weight,
            top_k=candidate_count,
        )
        candidates = [f.chunk for f in fused]

        wants_reranker = self.config.reranker.enabled if use_reranker is None else use_reranker
        if wants_reranker and self.reranker is not None and candidates:
            candidates = self.reranker.rerank(query, candidates)

        return candidates[:final_k]

    def close(self) -> None:
        self.vector_store.close()
