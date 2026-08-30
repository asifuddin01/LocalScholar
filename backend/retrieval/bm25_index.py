"""Lexical retrieval with BM25.

Dense retrieval is good at paraphrase and bad at rare literal strings, which is
exactly the wrong trade-off for research papers. "KiTS19", "SwinUNETR",
"Dice score", "AdamW" are the terms researchers actually search for, and an
embedding model that has never seen them maps them to a vague neighbourhood of
whatever looked similar in training. BM25 matches them exactly.

The index is held in memory and rebuilt from SQLite, which is the durable copy.
At laptop scale (a few thousand chunks) a rebuild is milliseconds, and it keeps
the two stores from drifting apart.
"""

from __future__ import annotations

import logging
import re
import threading

from rank_bm25 import BM25Okapi

from backend.ingestion.chunking import Chunk
from backend.retrieval.vector_store import ScoredChunk

logger = logging.getLogger(__name__)

# Keeps internal hyphens ("u-net", "fine-tuning") while still splitting on
# punctuation. Digits are kept because dataset names carry them: KiTS19, CIFAR-10.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, with hyphenated terms indexed both ways.

    "U-Net" is emitted as "u-net", "u" and "net" so it is found whether the
    query spells it "U-Net", "UNet" or "U Net". Used for both indexing and
    querying -- they must not diverge.
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text.lower()):
        tokens.append(match)
        if "-" in match:
            tokens.extend(part for part in match.split("-") if part)
    return tokens


class BM25Index:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Rebuild the whole index. Cheap at this scale, and always consistent."""
        with self._lock:
            self._chunks = chunks
            if not chunks:
                self._bm25 = None
                return
            self._bm25 = BM25Okapi([tokenize(c.search_text) for c in chunks])
            logger.info("BM25 index built over %d chunks", len(chunks))

    def search(
        self, query: str, top_k: int, document_ids: list[str] | None = None
    ) -> list[ScoredChunk]:
        with self._lock:
            if self._bm25 is None or not self._chunks:
                return []
            tokens = tokenize(query)
            if not tokens:
                return []
            scores = self._bm25.get_scores(tokens)
            chunks = self._chunks

        allowed = set(document_ids) if document_ids else None
        results = [
            ScoredChunk.from_chunk(chunk, float(score))
            for chunk, score in zip(chunks, scores)
            # A zero score means no query term appears at all. Returning those
            # to pad out top_k would hand the fusion stage pure noise.
            if score > 0 and (allowed is None or chunk.document_id in allowed)
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def __len__(self) -> int:
        return len(self._chunks)
