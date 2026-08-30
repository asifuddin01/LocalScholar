"""Measure embedding throughput and memory for the candidate models.

Written because the obvious default turned out to be the worst option on this
hardware. fastembed maps `BAAI/bge-small-en-v1.5` to an *int8-quantized* ONNX
build, and int8 kernels fall back to slow paths on Apple Silicon: it measured
4x slower than a non-quantized model of the same size. That is not something
you would guess from the model card, so it is measured here instead.

Usage:
    python scripts/benchmark_embeddings.py path/to/paper.pdf
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CANDIDATES = [
    "snowflake/snowflake-arctic-embed-xs",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en",
    "BAAI/bge-small-en-v1.5",
]

SAMPLE_SIZE = 48

# Each model runs in its own process so peak RSS is attributable to that model
# rather than to whatever ran before it.
WORKER = """
import sys, time, resource, json
from fastembed import TextEmbedding
model, path = sys.argv[1], sys.argv[2]
texts = json.load(open(path))
m = TextEmbedding(model_name=model)
list(m.embed(texts[:2]))                     # warm up: excludes lazy init
t0 = time.time(); vectors = list(m.embed(texts)); elapsed = time.time() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
print(json.dumps({"model": model, "seconds": elapsed, "rss_mb": rss,
                  "dim": len(vectors[0]), "n": len(texts)}))
"""


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    from backend.ingestion.chunking import chunk_document
    from backend.ingestion.pdf_parser import parse_pdf

    parsed = parse_pdf(sys.argv[1])
    chunks = chunk_document("bench", "bench.pdf", parsed)
    texts = [c.search_text for c in chunks][:SAMPLE_SIZE]
    print(f"{parsed.page_count} pages -> {len(chunks)} chunks; timing {len(texts)}\n")

    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "texts.json"
        corpus.write_text(json.dumps(texts))
        worker = Path(tmp) / "worker.py"
        worker.write_text(WORKER)

        print(f"{'model':44} {'chunks/s':>9} {'peak RSS':>10} {'dim':>5}")
        print("-" * 72)
        for model in CANDIDATES:
            result = subprocess.run(
                [sys.executable, str(worker), model, str(corpus)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"{model:44} {'FAILED':>9}  {result.stderr.strip().splitlines()[-1][:40]}")
                continue
            data = json.loads(result.stdout.strip().splitlines()[-1])
            rate = data["n"] / data["seconds"]
            print(f"{model:44} {rate:9.1f} {data['rss_mb']:9.0f}MB {data['dim']:5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
