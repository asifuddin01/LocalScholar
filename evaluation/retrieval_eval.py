"""Retrieval evaluation and the four-way ablation.

Answers a question the architecture diagram cannot: does any of this actually
help? BM25, dense, hybrid and hybrid+reranking are measured on the same
questions against the same index, so the numbers are comparable.

A chunk counts as relevant when it comes from the expected paper AND contains
every one of the question's `evidence_keywords`. That is stricter than
"the right paper" (which a one-paper library would trivially satisfy) and more
robust than pinning exact chunk ids, which would silently break the moment
chunking parameters change.

Usage:
    python evaluation/retrieval_eval.py            # all four methods
    python evaluation/retrieval_eval.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import load_config                      # noqa: E402
from backend.db import Catalogue                            # noqa: E402
from backend.retrieval.engine import RetrievalEngine        # noqa: E402
from backend.retrieval.vector_store import ScoredChunk      # noqa: E402

BENCHMARK_PATH = Path(__file__).parent / "dataset" / "benchmark.json"
K_VALUES = (1, 3, 5, 10)
RETRIEVE_K = max(K_VALUES)


@dataclass
class MethodResult:
    name: str
    recall_at: dict[int, float]
    mrr: float
    median_ms: float
    misses: list[str]


def is_relevant(chunk: ScoredChunk, question: dict, filename_by_id: dict[str, str]) -> bool:
    if filename_by_id.get(chunk.document_id) != question["paper"]:
        return False
    haystack = chunk.text.lower()
    return all(kw.lower() in haystack for kw in question["evidence_keywords"])


def validate_benchmark(questions: list[dict], catalogue: Catalogue) -> list[dict]:
    """Drop questions whose evidence isn't in the index, loudly.

    A question with no reachable gold chunk is unanswerable by construction:
    counting it as a miss would understate every method equally, and silently
    keeping it would make the benchmark a measure of the fixture rather than
    of retrieval. Either way the honest move is to report it and exclude it.
    """
    chunks = catalogue.get_chunks()
    filename_by_id = {c.document_id: c.filename for c in chunks}
    usable, unreachable = [], []

    for question in questions:
        if any(is_relevant(_as_scored(c), question, filename_by_id) for c in chunks):
            usable.append(question)
        else:
            unreachable.append(question["id"])

    if unreachable:
        print(
            f"WARNING: {len(unreachable)} question(s) have no matching chunk in the "
            f"index and were excluded: {', '.join(unreachable)}",
            file=sys.stderr,
        )
    return usable


def _as_scored(chunk) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk.chunk_id, document_id=chunk.document_id, filename=chunk.filename,
        title=chunk.title, text=chunk.text, page_number=chunk.page_number,
        section=chunk.section, score=0.0,
    )


def evaluate(engine: RetrievalEngine, questions: list[dict], method: str,
             filename_by_id: dict[str, str]) -> MethodResult:
    hits_at = {k: 0 for k in K_VALUES}
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    misses: list[str] = []

    for question in questions:
        started = time.perf_counter()
        if method == "bm25":
            results = engine.bm25_search(question["question"], top_k=RETRIEVE_K)
        elif method == "dense":
            results = engine.dense_search(question["question"], top_k=RETRIEVE_K)
        elif method == "hybrid":
            results = engine.hybrid_search(
                question["question"], top_k=RETRIEVE_K, use_reranker=False
            )
        elif method == "hybrid+rerank":
            results = engine.hybrid_search(
                question["question"], top_k=RETRIEVE_K, use_reranker=True
            )
        else:
            raise ValueError(method)
        latencies.append((time.perf_counter() - started) * 1000)

        rank = next(
            (i for i, c in enumerate(results, 1)
             if is_relevant(c, question, filename_by_id)),
            None,
        )
        if rank is None:
            misses.append(question["id"])
            reciprocal_ranks.append(0.0)
            continue
        reciprocal_ranks.append(1.0 / rank)
        for k in K_VALUES:
            if rank <= k:
                hits_at[k] += 1

    n = len(questions)
    return MethodResult(
        name=method,
        recall_at={k: hits_at[k] / n for k in K_VALUES},
        mrr=statistics.mean(reciprocal_ranks),
        median_ms=statistics.median(latencies),
        misses=misses,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="write raw results here")
    args = parser.parse_args()

    config = load_config()
    catalogue = Catalogue(config.storage.catalogue_path)
    engine = RetrievalEngine(config, catalogue)
    engine.rebuild_bm25()

    if len(engine.bm25) == 0:
        print("The index is empty. Upload papers before running the benchmark.",
              file=sys.stderr)
        return 1

    benchmark = json.loads(BENCHMARK_PATH.read_text())
    questions = validate_benchmark(benchmark["questions"], catalogue)
    filename_by_id = {c.document_id: c.filename for c in catalogue.get_chunks()}

    print(f"\n{len(questions)} questions over {len(engine.bm25)} chunks\n")
    header = f"{'method':16} " + " ".join(f"R@{k:<5}" for k in K_VALUES) + f"{'MRR':>7}{'median':>9}"
    print(header)
    print("-" * len(header))

    results = []
    for method in ("bm25", "dense", "hybrid", "hybrid+rerank"):
        result = evaluate(engine, questions, method, filename_by_id)
        results.append(result)
        cells = " ".join(f"{result.recall_at[k]:<7.3f}" for k in K_VALUES)
        print(f"{result.name:16} {cells}{result.mrr:7.3f}{result.median_ms:8.0f}ms")

    print("\nPer-method misses (question ids):")
    for result in results:
        print(f"  {result.name:16} {', '.join(result.misses) or 'none'}")

    if args.json:
        args.json.write_text(json.dumps({
            "n_questions": len(questions),
            "n_chunks": len(engine.bm25),
            "embedding_model": config.embeddings.model,
            "reranker_model": config.reranker.model,
            "results": [
                {"method": r.name, "recall_at": r.recall_at, "mrr": r.mrr,
                 "median_ms": r.median_ms, "misses": r.misses}
                for r in results
            ],
        }, indent=2))
        print(f"\nWrote {args.json}")

    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
