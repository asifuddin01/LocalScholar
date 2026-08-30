"""Combining two ranked lists into one.

Dense and BM25 scores are not comparable. Cosine similarity lives in roughly
[0.5, 0.8] for this corpus; BM25 is unbounded and routinely returns 6 for junk
and 18 for a good hit. Normalising them onto a shared scale and adding them up
means the fused ranking is dominated by whichever retriever happens to have the
wider spread on that particular query, which changes from query to query.

Reciprocal Rank Fusion sidesteps the problem by throwing the scores away and
using only the ranks:

    score(chunk) = sum over retrievers of  weight / (k + rank)

A chunk found by both retrievers beats one found by either alone, which is the
behaviour we actually want: agreement between a semantic and a lexical matcher
is strong evidence, and it is exactly what neither retriever achieves on its
own ("what optimizer was used?" is invisible to embeddings and obvious to BM25;
"what are the limitations?" is the reverse).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from backend.retrieval.vector_store import ScoredChunk

# The conventional RRF constant. Large enough that the top few ranks are close
# together, so a chunk ranked 1st by one retriever does not automatically beat
# a chunk ranked 2nd-and-3rd by both.
DEFAULT_RRF_K = 60


@dataclass
class FusedChunk:
    """A chunk plus where each retriever placed it. Kept for the ablation."""

    chunk: ScoredChunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None

    @property
    def found_by_both(self) -> bool:
        return self.dense_rank is not None and self.bm25_rank is not None


def reciprocal_rank_fusion(
    dense: list[ScoredChunk],
    bm25: list[ScoredChunk],
    *,
    k: int = DEFAULT_RRF_K,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    top_k: int | None = None,
) -> list[FusedChunk]:
    """Fuse two ranked lists. Ranks are 1-based."""
    fused: dict[str, FusedChunk] = {}

    for rank, chunk in enumerate(dense, start=1):
        fused[chunk.chunk_id] = FusedChunk(
            chunk=chunk, score=dense_weight / (k + rank), dense_rank=rank
        )

    for rank, chunk in enumerate(bm25, start=1):
        contribution = bm25_weight / (k + rank)
        existing = fused.get(chunk.chunk_id)
        if existing is None:
            fused[chunk.chunk_id] = FusedChunk(
                chunk=chunk, score=contribution, bm25_rank=rank
            )
        else:
            existing.score += contribution
            existing.bm25_rank = rank

    ordered = sorted(fused.values(), key=lambda f: f.score, reverse=True)
    if top_k is not None:
        ordered = ordered[:top_k]

    # Overwrite the per-retriever score with the fused one so downstream code
    # never accidentally compares a cosine value against a BM25 value.
    return [replace(f, chunk=replace(f.chunk, score=f.score)) for f in ordered]
