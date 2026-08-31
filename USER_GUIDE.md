# LocalScholar — User Guide

Everything you need to install, run and actually use LocalScholar day to day.
For the design rationale and benchmark results, see [README.md](README.md).

---

## Contents

1. [What you need](#1-what-you-need)
2. [Installation](#2-installation)
3. [Starting and stopping](#3-starting-and-stopping)
4. [Adding papers](#4-adding-papers)
5. [Asking questions](#5-asking-questions)
6. [Summarising a paper](#6-summarising-a-paper)
7. [Paper details](#7-paper-details)
8. [Comparing papers](#8-comparing-papers)
9. [Searching for evidence](#9-searching-for-evidence)
10. [Reading citations](#10-reading-citations)
11. [Configuration](#11-configuration)
12. [Troubleshooting](#12-troubleshooting)
13. [Your data and privacy](#13-your-data-and-privacy)
14. [FAQ](#14-faq)

---

## 1. What you need

| Requirement | Notes |
|---|---|
| macOS or Linux | Developed and tested on an Apple M1 with 8GB RAM |
| Python 3.11 or 3.12 | **Not 3.13+** — the ONNX runtime wheels lag behind |
| Node.js 18+ | For the web interface |
| Ollama | Runs the language model locally |
| ~4GB free disk | 1.9GB model, ~150MB embedding/reranking models, plus your papers |
| No API key | Nothing here talks to a paid service |

Check what you have:

```bash
python3.12 --version && node --version
```

---

## 2. Installation

**Step 1 — get the code and install Python dependencies.**

```bash
git clone https://github.com/asifuddin01/LocalScholar.git && cd LocalScholar
```

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

**Step 2 — install the web interface dependencies.**

```bash
npm --prefix frontend install
```

**Step 3 — install Ollama and pull the model.**

```bash
brew install ollama
```

Start the Ollama service (leave this running in its own terminal):

```bash
ollama serve
```

Then, in another terminal, download the model (~1.9GB, one time):

```bash
ollama pull qwen2.5:3b-instruct
```

**Step 4 — verify.**

```bash
.venv/bin/python -m pytest tests/ -q
```

You should see `88 passed`. If so, you are ready.

> The embedding and reranking models (~150MB) download automatically the first
> time you index a paper. After that, LocalScholar never needs the internet.

---

## 3. Starting and stopping

With `ollama serve` already running in its own terminal:

```bash
./scripts/dev.sh
```

Then open **<http://localhost:5173>**.

To stop, press `Ctrl+C` in that terminal.

If you prefer to run the two halves separately:

```bash
.venv/bin/python -m uvicorn backend.main:app --port 8000
```

```bash
npm --prefix frontend run dev
```

> **The first question you ask after starting takes ~30 seconds** while Ollama
> loads the model into memory. Every question after that takes about 2 seconds.
> LocalScholar warms the model up in the background at startup, so if you wait
> half a minute before your first question you will not notice this at all.

---

## 4. Adding papers

Drag PDFs onto the **Upload papers** box, or click it to pick files. You can
drop many at once.

Each paper then shows its status:

| Status | Meaning |
|---|---|
| **Processing…** | Being parsed and indexed. A 40-page paper takes ~25 seconds. |
| **Indexed** | Ready to use. Shows pages, chunks and detected sections. |
| **Failed** | Could not be used. The reason is shown underneath. |

Things worth knowing:

- **Duplicates are detected by content**, not filename. Re-uploading the same
  paper under a different name will not index it twice.
- **Scanned PDFs are rejected** with a clear message. LocalScholar does not run
  OCR, and indexing a paper as empty text would be worse than refusing it — it
  would look like it worked and then answer every question with "no evidence".
- **The section list is a good health check.** If a paper shows sensible
  sections (`Abstract`, `3.1 Dataset`, `IV. RESULTS`), parsing went well.
- Click any indexed paper to read the extracted text page by page. This is the
  fastest way to confirm a citation is pointing where you expect.
- The `×` button removes a paper and everything derived from it.

---

## 5. Asking questions

Type a question into **Ask your papers** and press Ask.

By default every paper is searched. Tick specific papers to narrow it — this
works properly: the filter is applied *inside* the search, so you still get a
full set of results from the papers you chose.

Questions that work well:

```
What dataset did they use, and how large was it?
What optimizer and learning rate were used?
What evaluation metrics did they report?
What limitations did the authors acknowledge?
What preprocessing was applied to the images?
How does their method differ from the baseline?
What are the common limitations across these papers?
```

Ask across several papers at once and the answer will attribute each claim to
the paper it came from.

### When it says it cannot answer

> *I could not find sufficient evidence in the uploaded papers to answer this
> question.*

This is a feature. It means the papers do not contain the answer, and
LocalScholar will not fill the gap from the language model's general knowledge.
If you see this and believe the paper *does* say something:

- Check the paper is ticked (or that nothing is ticked, to search everything)
- Try the paper's own vocabulary — "Dice score" rather than "how accurate"
- Use **Search your papers** (section 9) with **Keyword** mode to find the exact
  term, which tells you whether the wording exists in the paper at all

---

## 6. Summarising a paper

Tick **exactly one** paper, then ask:

```
summarise this
```

Other phrasings that work: *summarize this paper*, *give me an overview*,
*what is this paper about*, *key points*, *tl;dr*.

You get a structured summary with a citation on each section:

- Research problem
- Main contribution
- Data
- Methodology
- Experimental setup
- Results
- Limitations
- Future work

The first summary of a paper takes about a minute; it is cached, so asking
again is instant.

> If more than one paper is in scope, LocalScholar asks you to pick one rather
> than blending several papers into a single summary that describes none of
> them accurately.

---

## 7. Paper details

Open **Paper details & comparison** and click a paper's button. You get a table
of fifteen extracted fields — title, authors, year, research problem, task,
dataset, dataset size, model/architecture, method, training procedure, loss
function, evaluation metrics, main results, limitations, future work.

Expand **supporting excerpts** at the bottom to see the passages every value
was drawn from.

**"Not reported" is a real answer.** It means no retrieved passage supported a
value. Sometimes the paper genuinely does not say; sometimes the field does not
apply (a genomics paper has no "loss function"). Either way, LocalScholar will
not invent one. Expect a properly filled table to still have empty rows.

The first extraction takes 50–90 seconds per paper and is cached afterwards.

---

## 8. Comparing papers

Tick two or more papers, then click **Compare N selected papers**.

You get a side-by-side table across dataset, dataset size, task, architecture,
training procedure, evaluation metrics, main results and limitations.

Comparison reuses each paper's cached details, so the first comparison of
papers you have never opened will run the extraction for each (about a minute
each). After that it is instant.

---

## 9. Searching for evidence

**Search your papers** returns raw passages instead of a written answer. Use it
when you want to see the evidence yourself, or to check whether a term appears
in your library at all.

Two modes, and the difference is genuinely useful:

| Mode | Good at | Bad at |
|---|---|---|
| **Semantic** | Paraphrase — "how many samples" finds "13,599 images" | Rare exact strings it has never seen |
| **Keyword** | Exact terms — `AdamW`, `KiTS19`, `F1-score` | Anything phrased differently |

If Keyword mode returns nothing for a term, that term is genuinely not in your
papers. That is often the fastest way to settle a question.

---

## 10. Reading citations

Every claim carries a numbered marker like `[1]`. Click it to jump to the
passage it came from, showing the **paper, page number and section**.

Two guarantees are enforced by the code, not by asking the model nicely:

1. **A citation always points at a real retrieved passage.** If the model
   invents `[7]` when only six passages exist, it is removed before you see it.
2. **An answer nothing supports is not shown.** Instead you get the
   "insufficient evidence" message.

For extracted fields and comparison cells, citations are computed by matching
the extracted value back against the source text — so a citation there is a
check that the paper really says it, not the model's word for it.

To verify anything yourself: click the paper in **Your papers** and read the
page the citation names.

---

## 11. Configuration

Everything lives in `configs/default.yaml`. Restart the backend after editing.

### Using a better model

If you have 16GB RAM or more, this is the single highest-value change:

```yaml
llm:
  model: qwen2.5:7b-instruct
```

Then `ollama pull qwen2.5:7b-instruct`. It is noticeably better at
multi-paper synthesis and at filling extraction fields.

### Making it faster

```yaml
reranker:
  enabled: false      # saves ~1.6s per question, costs some accuracy
```

### Retrieval tuning

```yaml
retrieval:
  dense_top_k: 20     # candidates from embedding search
  bm25_top_k: 20      # candidates from keyword search
  final_top_k: 6      # passages the model actually reads
```

Raise `final_top_k` if answers feel like they are missing context; lower it if
answers wander.

### Chunking

```yaml
chunking:
  chunk_size: 800     # characters per passage
  overlap: 120
```

Chunks never cross a page or section boundary regardless of these numbers —
that is what keeps one page number attached to every citation.

### Using a remote model instead

```yaml
llm:
  provider: openai
  base_url: http://localhost:8001/v1
  model: your-model
```

For a local vLLM or LM Studio server. **Setting this means document text leaves
this machine** if `base_url` is not local. Nothing ever falls back to a remote
provider automatically — if your local model is down, requests fail loudly.

---

## 12. Troubleshooting

### "Cannot reach Ollama at http://localhost:11434"

Ollama is not running. In a separate terminal:

```bash
ollama serve
```

Retrieval and search keep working without it; only answering needs a model.

### "Model 'qwen2.5:3b-instruct' is not installed"

```bash
ollama pull qwen2.5:3b-instruct
```

### The first question takes 30 seconds

Expected — Ollama is loading the model. Subsequent questions take ~2s. The
model stays resident for 30 minutes of inactivity (`llm.keep_alive`).

### A paper is stuck on "Processing…"

Restart the backend. LocalScholar detects documents orphaned by a restart and
re-queues them automatically on startup. (Parsing runs in a background task,
and background tasks do not survive a restart.)

### "It is probably a scanned document"

The PDF has no selectable text. LocalScholar does not do OCR. Run the file
through an OCR tool first (macOS Preview, `ocrmypdf`) and upload the result.

### Answers say "insufficient evidence" for things the paper says

Try the paper's own wording, and use **Keyword** search to check the term
exists. If Keyword search finds the passage but answering does not use it,
raise `retrieval.final_top_k`.

### Search results look wrong after deleting a paper

The vector index can lose sync with the database when documents are removed.
Restart the backend — the index is health-checked at startup and rebuilt
automatically from the database, which always holds the text.

### "Storage folder ... is already accessed by another instance"

Embedded Qdrant allows one process at a time. **Stop the backend before running
the evaluation scripts**, and restart it afterwards.

### Everything is slow / the machine is swapping

On 8GB, the model plus embedding and reranking models is a tight fit. Set
`reranker.enabled: false`, and close other heavy applications. Do not switch to
a 7B model on 8GB.

### Tests print a `recursive_mutex` error at the end

Harmless. It is a destructor race between native libraries at interpreter exit
and happens *after* results are reported. Check the `88 passed` line.

---

## 13. Your data and privacy

- PDFs are copied into `data/uploads/` and never leave your machine.
- Extracted text, chunks, summaries and extractions live in
  `data/localscholar.db` (SQLite). Vectors live in `data/qdrant/`.
- `data/` is gitignored, so a library cannot be committed by accident.
- The only network access is the one-time model downloads. Indexing and
  answering make no network calls at all — you can pull the network cable and
  everything still works.
- To delete everything: remove the `data/` directory.
- To back up your library: copy the `data/` directory.

---

## 14. FAQ

**Can I use this without Ollama?**
Partly. Uploading, indexing and both search modes work. Answering, summarising,
details and comparison need a model.

**How many papers can it handle?**
Tested with ~1000 chunks (six papers) with sub-10ms retrieval. BM25 is rebuilt
in memory at startup, so very large libraries will slow startup before they
slow searching. A few hundred papers should be fine on a laptop.

**Why is extraction so much slower than answering?**
Answering is one model call. Extraction is five, each with its own retrieval.
It is cached, so you pay once per paper.

**Can it read tables and figures?**
Table *text* is extracted and searchable. Figure images are not — there is no
vision model in the pipeline. A number that exists only inside a chart image
cannot be found.

**Why does it refuse to answer things I know are in the paper?**
Most often the paper phrases it differently from your question. Use Keyword
search to find the paper's own wording, then ask using that.

**Can I add papers while it is running?**
Yes. Upload at any time; indexing happens in the background and the interface
updates itself.

**Does it remember previous questions?**
No. Each question is independent — follow-ups like "and how large was it?" will
not resolve. Ask complete questions.

**How do I reset a paper's summary or details?**
Delete the paper and re-upload it, or call the API with `?refresh=true`.
