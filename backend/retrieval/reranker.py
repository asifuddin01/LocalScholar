"""Cross-encoder reranking.

Dense retrieval compares two vectors that were produced without ever seeing
each other. A cross-encoder reads the query and the chunk *together* and scores
the pair directly, which is much more accurate and much too slow to run over a
whole corpus -- so it only ever sees the couple of dozen candidates that
survived fusion. Retrieve broadly, then rank precisely.

Optional by design: if the model can't be downloaded or fails to load, the
pipeline logs it once and carries on with the fused ranking rather than
breaking the application.
"""

from __future__ import annotations

import logging
import threading

from backend.retrieval.vector_store import ScoredChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        cache_dir: str | None = None,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self._cache_dir = cache_dir
        self._batch_size = max(1, batch_size)
        self._model = None
        self._failed = False
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None and not self._failed:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            logger.info("Loading reranker %s", self.model_name)
            self._model = TextCrossEncoder(
                model_name=self.model_name, cache_dir=self._cache_dir
            )
        return self._model

    def rerank(self, query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        """Re-score and re-order candidates. Returns them unchanged on failure."""
        if not chunks or self._failed:
            return chunks
        try:
            with self._lock:
                model = self._load()
                # Same small-batch constraint as the embedder: onnxruntime
                # deadlocks on large batches off the main thread on macOS/ARM.
                scores = list(
                    model.rerank(
                        query,
                        [c.text for c in chunks],
                        batch_size=self._batch_size,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - reranking is a bonus, not a requirement
            self._failed = True
            logger.warning("Reranker unavailable, continuing without it: %s", exc)
            return chunks

        from dataclasses import replace

        rescored = [replace(c, score=float(s)) for c, s in zip(chunks, scores)]
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored
