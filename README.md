# LocalScholar

A research-paper assistant that runs entirely on your own machine. No API key, no uploads,
no account. Ask questions across your PDF library and get answers with page-level citations
you can check.

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/code-MIT-green)

---

## Why this exists

Reading a paper the first time is interesting. Going back a fourth time to re-find which
dataset they used, how many samples it had, and what the authors admitted in the limitations
section is not. That re-reading is most of the cost of a literature review, and it scales
badly: comparing eight papers on those axes means forty separate searches through eight PDFs.

LocalScholar indexes papers locally and answers those questions with citations down to the
page and section, so every claim can be checked against the PDF. It runs on a local LLM
because papers under review are often unpublished, confidential, or someone else's
intellectual property, and pasting them into a hosted API is frequently not an option.

## What it does

- **Multi-PDF library** — drag in a stack of papers; each is parsed, chunked and indexed
- **Hybrid retrieval** — dense embeddings + BM25, fused with Reciprocal Rank Fusion
- **Cross-encoder reranking** — a second, more accurate pass over the fused candidates
- **Grounded answers** — every claim carries a `[n]` citation you can click through to the
  supporting passage, with paper, page and section
- **Refuses to guess** — if the papers don't support an answer, it says so
- **Structured extraction** — dataset, size, architecture, training, metrics, results,
  limitations, future work, each with its own citations
- **Paper comparison** — the same extraction rendered as a side-by-side table
- **Runs offline** — after the one-time model downloads, no network access at all

## Quick start

Requires **Python 3.11 or 3.12** (not 3.13+; the ONNX wheels lag), **Node 18+**, and
**[Ollama](https://ollama.com)**.

```bash
git clone <repo-url> && cd LocalScholar
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
npm --prefix frontend install
```

Get the local model (~1.9GB, once):

```bash
brew install ollama && ollama serve
```

```bash
ollama pull qwen2.5:3b-instruct
```

Then start both processes:

```bash
./scripts/dev.sh
```

Open <http://localhost:5173> and drop in some PDFs. (Or run them separately:
`.venv/bin/python -m uvicorn backend.main:app --port 8000` and
`npm --prefix frontend run dev`.)

The embedding and reranking models (~150MB total) download from Hugging Face on first use and
are cached in `data/models`. After that the whole system runs with no network access.

## Architecture

```
   PDF ──► parse ──► chunk ──► embed ──────► Qdrant  (dense, on disk)
            │          │                 └─► BM25    (lexical, in memory)
            │          │
            │          └─ every chunk keeps: document, page, section
            │
            └─ two-column reading order, heading detection, header stripping


   question
      │
      ├──► dense retrieval  (top 20) ──┐
      │                                ├─► Reciprocal Rank Fusion ─► top 24
      └──► BM25 retrieval   (top 20) ──┘                              │
                                                                      ▼
                                                        cross-encoder reranker
                                                                      │
                                                                 top 6 chunks
                                                                      ▼
                                             local LLM (Ollama) with numbered sources
                                                                      │
                                                    answer + validated citations
```

Nothing leaves the machine at any stage.

## Why each component is there

**Two retrievers, because they fail in opposite directions.** Embeddings are good at
paraphrase and bad at rare literal strings; BM25 is the reverse. On this corpus, "what
optimizer was used?" is nearly invisible to the embedding model and lands BM25 directly on the
hyperparameter table containing `AdamW`. "What are the limitations?" is the other way round.
Neither is sufficient alone, and the ablation below shows it.

**Reciprocal Rank Fusion rather than adding scores.** Cosine similarity sits in roughly
[0.5, 0.8] here; BM25 is unbounded and routinely returns 6 for junk and 18 for a good hit.
Normalising and summing lets whichever retriever happens to have the wider spread on that
query dominate. RRF discards the scores and uses only ranks, so a chunk found by *both*
retrievers beats one found by either alone.

**A cross-encoder reranker on the shortlist only.** Dense retrieval compares two vectors
produced without ever seeing each other. A cross-encoder reads the query and the passage
together and is far more accurate — and far too slow to run over a whole corpus. Retrieve
broadly, then rank precisely. It is the single biggest win in the ablation (+0.10 Recall@1)
and also the slowest stage (~2.5s), so it is one config flag away from being switched off.

**Chunks never cross a page or a section.** This splits the occasional paragraph, and it is
worth it: it means every chunk has exactly one page number. A chunk straddling pages 4 and 5
can only ever produce a vague citation, and a vague citation is one nobody can check.

**SQLite + embedded Qdrant, no services.** A vector database you have to start is a vector
database that makes `git clone && run` fail.

## Evaluation

30 hand-written questions over four real papers (a Springer/LNCS retinal-imaging paper, an
IEEE food-expiry paper, a 34-page IEEE federated-learning survey, and a 43-page Nature
CellOracle paper) — 705 chunks in total. Questions cover datasets, methods, results,
limitations, extraction and bare terminology.

A chunk counts as relevant only if it comes from the expected paper **and** contains the
question's evidence keywords. The harness validates every question against the live index
first and loudly excludes any whose evidence is unreachable, so an unanswerable question can
never quietly inflate a score.

Reproduce with:

```bash
.venv/bin/python evaluation/retrieval_eval.py
```

### Retrieval ablation

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Median latency |
|---|---|---|---|---|---|---|
| BM25 only | 0.533 | 0.767 | 0.833 | 0.867 | 0.659 | 1 ms |
| Dense only | 0.733 | 0.833 | 0.867 | 0.900 | 0.795 | 7 ms |
| Hybrid (RRF) | 0.700 | 0.900 | 0.900 | 0.933 | 0.798 | 10 ms |
| **Hybrid + reranker** | **0.800** | **0.900** | **0.933** | **0.933** | **0.857** | 2517 ms |

Measured, not estimated. Two things in that table are worth reading carefully:

- **Hybrid's Recall@1 (0.700) is *below* dense alone (0.733).** Fusion clearly helps deeper in
  the ranking — Recall@3 goes 0.833 → 0.900 — but mixing in BM25's ordering can knock the
  single best chunk off the top spot. This is exactly the gap the reranker closes, taking
  Recall@1 to 0.800. Reporting the hybrid row without the reranker row would have made hybrid
  look like a regression; reporting only the final number would have hidden why the reranker
  is there.
- **The reranker costs ~250x the latency of fusion** for +0.06 MRR. On this hardware that is
  ~2.5s per question against ~2.2s for the model itself. Worth it here, and `reranker.enabled:
  false` for anyone who disagrees.

Two questions are missed by every method (`d5`, `e6`), which is a fair result rather than a
tuning failure: both ask about a concept the paper expresses in wording no retriever matches.

### Refusing to answer

The benchmark also carries four questions the library provably cannot answer — a dataset no
paper uses, a cost never discussed, a physics fact, and a task outside the corpus.

**4 / 4 correctly refused.** The system replies "I could not find sufficient evidence in the
uploaded papers" rather than producing a fluent answer from the model's own memory. This is
the property that decides whether the tool is usable for real literature review: a tool that
quietly answers from pretraining when the paper is silent is worse than no tool, because you
cannot tell the two cases apart.

### Answer quality

Answers were checked by reading them against the pages they cite. Representative results:

| Question | Answer | Cited |
|---|---|---|
| How many fruit images, and how were they split? | "13,599 fruit images... 10,901 training and 2,698 testing" | p2, *A. Dataset Description* ✓ |
| What optimizer and learning rate were used? | "AdamW... 5e-5" | p13, *6 Results and Discussion* ✓ |
| What are the limitations acknowledged? | the paper's own limitation paragraph | p4, *F. Limitation* ✓ |

Faithfulness was verified by hand rather than by an LLM judge — with a four-paper corpus, a
judge would have added a second source of error without adding trustworthy signal. That is a
gap, and it is listed as one below.

## Measurements that changed the design

Three findings that came from running against real papers rather than reasoning about them.

**The obvious embedding model was the worst option.** fastembed serves
`BAAI/bge-small-en-v1.5` as an *int8-quantized* ONNX build, and int8 kernels fall back to slow
paths on Apple Silicon:

| Model | chunks/s | Peak RSS |
|---|---|---|
| `BAAI/bge-small-en-v1.5` (quantized) | 7.4 | 720 MB |
| `BAAI/bge-small-en` (not quantized) | 12.9 | 569 MB |
| **`snowflake/snowflake-arctic-embed-xs`** | **30.8** | 654 MB |
| `all-MiniLM-L6-v2` | 77.8 | 387 MB |

Reproduce with `python scripts/benchmark_embeddings.py <paper.pdf>`.

**onnxruntime deadlocks off the main thread.** Indexing runs in a background thread so uploads
stay responsive. With onnxruntime's default batch of 256, `session.run()` hangs *silently* —
no exception, no timeout, the document sits at "processing" forever while the process idles at
2% CPU. Batches of 128/64/32/16 all work. The default is 32, and there is a regression test
that indexes a 490-chunk document through the real background-thread path.

**`query_embed()` does nothing.** fastembed exposes a query-side method that looks like it
applies the model's instruction prefix; for these models it is a plain alias for `embed()`,
verified by comparing the two vectors. LocalScholar applies the prefix itself. Before the fix,
"What are the limitations of this approach?" retrieved an IEEE copyright banner; after it, the
paper's actual `F. Limitation` section.

## How grounding is enforced

Refusing to fabricate is a property of the pipeline, not a line in the prompt:

1. The model only ever sees text retrieved from your PDFs.
2. Every passage is numbered and the model must cite those numbers.
3. **Citations pointing at a source that wasn't retrieved are stripped after generation**, so
   an invented `[7]` cannot reach you.
4. **An answer with no surviving citation is rejected** and reported as no-evidence.
5. If nothing is retrieved, the model is never called at all.

Point 4 earns its keep. Asked to extract paper metadata, the 3B model confidently produced the
title *"Privacy-Preserving Ordinal-Meta Learning for Food Freshness"* by *"Kumar et al. (2025)"*
for a paper actually called *"Leveraging CNN and Random Forest for Accurate Food Expiry
Prediction"*. It cited nothing, because nothing in the paper said it, and the rule caught all
three fabricated fields automatically. The title is now read off page one by the parser
instead: never ask a model for something you already know.

### Citations are computed, not self-reported

For structured extraction the rule goes further, and this was the most useful thing the build
taught me.

The first design asked the model to return `{"value": ..., "citations": [1, 2]}` per field. A
3B model ignores that shape and returns plain strings, so every field arrived with an empty
citation list and got discarded — the extraction table came back almost empty even though the
evidence was sitting in the retrieved excerpts. The obvious fix was to nag the prompt harder.

The better fix was to stop asking. **A citation the model reports is a claim about its own
reasoning; a citation computed by matching the extracted value back against the excerpt text
is a check on the output.** So the model now returns bare values, and each one is verified by
token overlap against the passages it was given. A value no passage supports is not reported,
whatever the model asserted.

That single change took extraction from 6/15 fields to 12/15 on one paper and 4/15 to 10/15 on
another, while making every remaining value *more* trustworthy rather than less — because now
the citation is evidence rather than testimony.

Fields the papers genuinely don't state come back as **"Not reported"**. A comparison table
with every cell filled in is a table that has been guessed at.

## The parser

Citations are only as good as the metadata attached to the text, so every block keeps its page
number and section. Four problems had to be solved, each found on a real paper:

- **Two-column reading order.** Sorting blocks top-to-bottom reads a two-column paper as "left
  line 1, right line 1, left line 2…", shredding every sentence.
- **Spaces.** PyMuPDF emits inter-word spaces as their own spans; filtering spans by "has
  content" before joining produces `LeveragingCNNandRandomForest`.
- **Headings vs. table cells.** Font size and weight are useless signals on their own — the
  IEEE template sets *every* heading at body size with no bold. Detection uses layout instead:
  a heading owns its line, a table cell has siblings beside it at the same height. A false
  heading is worse than a missed one, because it becomes the running section label for
  everything after it.
- **Split section numbers.** Springer/LNCS stores `1  Introduction` as two text lines; the
  stray `1` looks like a table cell and suppresses the heading. Rejoining them took one paper
  from 3 detected sections to 20.

Running headers are stripped before indexing — `Title Suppressed Due to Excessive Length 15`
and a Nature volume footer were both observed outranking real content for "what are the
limitations?".

Section recovery across three templates:

| Paper | Template | Pages | Sections |
|---|---|---|---|
| Retinal Fundus Multi-Disease Classification | Springer/LNCS, 1-col | 17 | 20 — `1 Introduction` → `7 Conclusion` |
| Food Expiry Prediction | IEEE, 2-col | 4 | 17 — including every `A.`–`H.` subsection |
| Federated Learning in Mobile Edge Networks | IEEE survey, 2-col | 34 | 37 — matches the paper's table of contents |

## Configuration

Everything tunable lives in `configs/default.yaml`; only keys the code actually reads are
listed there. Highlights:

```yaml
chunking:   { chunk_size: 800, overlap: 120 }
embeddings: { model: snowflake/snowflake-arctic-embed-xs, batch_size: 32 }
retrieval:  { dense_top_k: 20, bm25_top_k: 20, final_top_k: 6, rrf_k: 60 }
reranker:   { enabled: true }
llm:        { provider: ollama, model: qwen2.5:3b-instruct, keep_alive: 30m }
```

`qwen2.5:3b-instruct` is the default because it fits in 8GB alongside the embedding and
reranking models. On 16GB+, `qwen2.5:7b-instruct` is noticeably better at multi-paper
synthesis — change one line.

For an OpenAI-compatible server, set `llm.provider: openai` and `llm.base_url`. Nothing ever
falls back to it automatically: if the local model is down, requests fail loudly. Silently
shipping an unpublished paper to a hosted API because a local server was off would break the
one promise this project makes.

## Performance on an 8GB M1

| Operation | Time |
|---|---|
| Parse + index a 43-page paper | ~25 s |
| Retrieval (hybrid, no reranker) | ~10 ms |
| Retrieval + reranking | ~2.5 s |
| Answer generation (model warm) | ~2.2 s |
| First question after startup | ~30 s (model load) |
| Structured extraction, one paper | ~27–50 s (cached afterwards) |

The model is warmed at startup and held resident via `keep_alive`, because a cold call costs
~32s of loading against ~2.2s warm for identical work.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q        # 74 tests, ~5s
.venv/bin/python -m pytest tests/ -m slow   # + real embedding model
```

Covers PDF parsing, page/section preservation, column ordering, heading classification against
the real false positives that motivated each rule, chunking invariants, BM25 and vector
search, citation validation, and the API surface. The slow tests exercise the real model,
including the 490-chunk background-thread indexing regression test.

Citation handling has its own tests, because it is the feature most worth protecting:
invented citations are stripped, an answer left with none is rejected, and extracted values
are only cited when the source text actually contains them.

Test PDFs are generated in memory, so the layout under test is readable in the test file
rather than hidden in a committed binary.

## Limitations

Honest ones:

- **No OCR.** Scanned PDFs are rejected with a clear message rather than indexed as empty — a
  document that silently indexes to nothing looks successful and then answers everything with
  "no evidence found".
- **A 3B model is a 3B model.** It is reliable at extraction and citation-following, and
  visibly weaker at synthesising across four papers at once. `qwen2.5:7b-instruct` is a
  one-line change on a machine with the RAM for it.
- **Structured extraction takes ~55s per paper** on this hardware. Cached after the first run,
  but the first comparison of four papers is a few minutes.
- **Two of thirty benchmark questions are missed by every retrieval method.**
- **The evaluation corpus is four papers.** Enough to catch real bugs and rank methods
  against each other, not enough to claim a general result.
- **Generation quality is not scored automatically.** Retrieval has Recall@K and MRR;
  faithfulness was checked by reading answers against the cited pages, not by an LLM judge.
- Full-width figures in a two-column paper are read after both columns rather than in place.
  Page numbers are unaffected, so citations stay correct.
- An intermittent `recursive_mutex` abort can appear at interpreter exit after the test suite
  finishes — a destructor race between the native extensions. It happens after results are
  reported and does not affect them.

## Not built, on purpose

Scope was cut deliberately to land a working system rather than a broad, half-finished one:
Docker, response streaming, conversation history with follow-up query rewriting, and an
LLM-judged faithfulness score. Each is a real feature; none of them is the difference between
this working and not working.

## Project layout

```
backend/
  ingestion/   pdf_parser.py, chunking.py
  retrieval/   embeddings, vector_store, bm25_index, fusion, reranker, engine
  generation/  llm.py, answering.py, extraction.py
  api/         documents, search, ask, research
frontend/      React + Vite
evaluation/    benchmark.json, retrieval_eval.py
scripts/       benchmark_embeddings.py
tests/         74 tests
```

## License

Project code: MIT.

**PyMuPDF is AGPL-3.0**, which is the binding constraint if you redistribute this. It was
chosen anyway because extraction quality on multi-column scientific PDFs is what everything
else rests on. Swapping in `pypdf` (BSD) is possible — the parser is isolated behind
`parse_pdf()` — at a real cost in section detection and reading order.
