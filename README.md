# LocalScholar

A privacy-first research paper assistant that runs entirely on your own machine — no API key, no uploads to anyone's server.

> **Status: Milestone 1 of 7 complete.** The library, the PDF pipeline and the UI work end to end today.
> Retrieval, the local LLM and citations land in the following milestones. See [Roadmap](#roadmap).
> Nothing in this README describes a feature that isn't implemented.

---

## Why this exists

Reading a paper for the first time is interesting. Going back to it for the fourth time to re-find
which dataset they used, how many samples were in it, and what the authors admitted in the
limitations section is not. That re-reading is most of the cost of a literature review, and it
scales badly: comparing eight papers on those axes means forty separate searches through eight PDFs.

LocalScholar indexes your papers locally and answers those questions with page-level citations, so
every claim can be checked against the actual PDF. It runs against a local LLM because papers under
review are frequently unpublished, confidential, or someone else's intellectual property, and
pasting them into a hosted API is often not an option.

## What works today

| Capability | Status |
|---|---|
| Upload many PDFs at once, drag-and-drop | ✅ |
| Local PDF parsing with page + section metadata | ✅ |
| Two-column research-paper layout handling | ✅ |
| Automatic section outline extraction | ✅ |
| Content-addressed deduplication | ✅ |
| Per-file error reporting, background processing | ✅ |
| Inspect extracted text page by page | ✅ |
| Chunking, embeddings, vector + BM25 index | Milestone 2 |
| Hybrid retrieval, reranking, grounded QA | Milestones 3–4 |
| Summaries, structured extraction, comparison | Milestone 5 |
| Evaluation benchmark and retrieval ablation | Milestone 6 |
| Docker | Milestone 7 |

## Quick start

Requires **Python 3.11 or 3.12** (not 3.13+ — the ONNX runtime wheels lag) and **Node 18+**.

```bash
git clone <repo-url> && cd LocalScholar
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
npm --prefix frontend install
```

Run the two processes in separate terminals:

```bash
.venv/bin/python -m uvicorn backend.main:app --reload --port 8000
```

```bash
npm --prefix frontend run dev
```

Then open <http://localhost:5173>. Drop some PDFs in.

No API key is needed, and no network call is made while indexing.

## Stack, and why each piece is there

Every dependency here is carrying weight. The notable choices:

**PyMuPDF for parsing.** Research PDFs are two-column, ligature-heavy and full of tables. PyMuPDF
exposes per-span font size, weight and bounding boxes, which the parser needs to tell a section
heading from a table cell. Pure-text extractors throw that information away.

**fastembed (ONNX) instead of sentence-transformers.** Both run the same BGE embedding models, but
sentence-transformers pulls in PyTorch: about 2.5GB installed. fastembed runs the same models on
`onnxruntime` in roughly 300MB total, which matters a great deal when a local LLM is already
competing for memory on an 8GB laptop. It also supplies the cross-encoder reranker, so the reranking
stage needs no extra ML framework.

**Qdrant in embedded mode.** Runs in-process against a local directory. No server, no container, no
port. A vector database you have to start is a vector database that makes `git clone && run` fail.

**SQLite for the catalogue.** The library metadata and extracted page text are relational and tiny.

**FastAPI + React/Vite.** Background tasks come free with FastAPI, which is what keeps upload
responsive while a 34-page survey is parsed.

## Architecture (as built)

```
      ┌──────────────┐   multipart    ┌─────────────────────┐
      │  React UI    │ ─────────────► │  FastAPI            │
      │  :5173       │ ◄───────────── │  :8000              │
      └──────────────┘   poll status  └──────────┬──────────┘
                                                 │
                              validate + SHA-256 │  (duplicate check)
                                                 ▼
                                      ┌─────────────────────┐
                                      │ data/uploads/*.pdf  │
                                      └──────────┬──────────┘
                                                 │  background task
                                                 ▼
                                      ┌─────────────────────┐
                                      │ pdf_parser          │
                                      │  · column ordering  │
                                      │  · heading tracking │
                                      │  · de-hyphenation   │
                                      └──────────┬──────────┘
                                                 ▼
                                      ┌─────────────────────┐
                                      │ SQLite catalogue    │
                                      │  documents / pages  │
                                      └─────────────────────┘
```

## The parser is the interesting part

Citations are the whole point of this project, and a citation is only as good as the metadata
attached to the text it came from. Every block the parser emits carries `page_number` and `section`.
If those are lost at parse time, no amount of retrieval quality gets them back.

Four problems had to be solved to keep them accurate, each found by running against real papers:

**Two-column reading order.** Sorting blocks top-to-bottom reads a two-column paper as "left line 1,
right line 1, left line 2…", which shreds every sentence. The parser detects the two-column case
from block geometry and reads each column to its end.

**Spaces.** PyMuPDF emits inter-word spaces as their own spans. Filtering spans by "does it have
content" before joining them produces `LeveragingCNNandRandomForest` — which destroys tokenisation
for BM25 as thoroughly as it destroys readability.

**Headings vs. table cells.** Detection is biased hard toward precision, because a false heading is
worse than a missed one: it becomes the running section label for everything after it, so a single
bold table cell can mislabel half a page and every citation drawn from it. Font size and weight turn
out to be useless signals on their own — the IEEE template sets *every* heading at body size with no
bold — so the parser uses layout instead: a heading owns its line, while a table cell has siblings
to its left and right at the same height. Reference lists switch heading detection off entirely.

**Split section numbers.** Springer/LNCS papers store `1  Introduction` as two separate text lines.
Those are rejoined before anything else runs, which fixes both the lost numbering and the heading
that the stray `1` was suppressing.

Measured against three real papers in three different templates:

| Paper | Template | Pages | Sections recovered |
|---|---|---|---|
| Retinal Fundus Multi-Disease Classification | Springer/LNCS, 1-col | 17 | 20 — full outline, `1 Introduction` → `7 Conclusion` |
| Food Expiry Prediction (CNN + RF) | IEEE, 2-col | 4 | 17 — including all `A.`–`H.` subsections |
| Federated Learning in Mobile Edge Networks | IEEE survey, 2-col | 34 | 37 — matches the paper's table of contents |

## Configuration

`configs/default.yaml` holds the tunable settings. Only keys the code actually reads are listed
there; the file grows with each milestone.

```yaml
storage:
  data_dir: ./data
ingestion:
  max_upload_mb: 50
```

Two optional environment overrides (see `.env.example`): `LOCALSCHOLAR_CONFIG` and
`LOCALSCHOLAR_DATA_DIR`.

Heuristic constants in the parser are deliberately *not* exposed as configuration — changing them
requires reading the code, and surfacing them as user settings would imply a tuning story that
doesn't exist.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

30 tests, covering PDF parsing, page-number and section preservation, column ordering, heading
classification against the real false positives that motivated each rule, and the API surface
(multipart upload, background status transitions, deduplication, per-file rejection, deletion).

Test PDFs are generated in memory rather than committed as binaries, so the layout under test is
readable in the test file itself.

## Privacy

Uploaded PDFs are written to `./data/uploads` and never leave the machine. Indexing makes no network
calls. `data/` is gitignored, so a library cannot be committed by accident.

Once the embedding and reranking models arrive in Milestone 2, they are downloaded from Hugging Face
**once** and cached locally; after that first download the system runs fully offline. Document
content is never part of that request.

## Known limitations

- **No OCR.** Scanned, image-only PDFs are rejected with a clear message rather than indexed as
  empty. This is deliberate: a document that silently indexes to nothing looks successful and then
  answers every question with "no evidence found".
- **Author blocks on title pages** can be ordered oddly when a paper uses three side-by-side author
  columns. Body text is unaffected.
- **Title extraction** falls back to the filename for papers whose first page is dominated by a
  publisher copyright banner.
- Full-width figures and tables in a two-column paper are read after both columns rather than at
  their exact position. Page numbers are unaffected, so citations stay correct.

## Roadmap

1. ✅ **Foundation** — repo, backend, frontend, upload, PDF parsing
2. **Indexing** — chunking, embeddings, Qdrant, BM25
3. **RAG** — hybrid retrieval, context construction, local LLM via Ollama
4. **Citations** — page-level grounding, evidence display, "not found" behaviour, reranking
5. **Research features** — summaries, structured extraction, multi-paper questions, comparison
6. **Evaluation** — benchmark, Recall@K, MRR, faithfulness, retrieval ablation
7. **Polish** — Docker, error handling, README, screenshots

## License

Project code: MIT.

Note that **PyMuPDF is AGPL-3.0**, which is the binding constraint if you redistribute this.
It was chosen anyway because extraction quality on multi-column scientific PDFs is the foundation
everything else rests on. Swapping it for `pypdf` (BSD) is possible — the parser is isolated behind
`parse_pdf()` — at a real cost in section detection and reading order.
