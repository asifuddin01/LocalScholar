"""Grounded question answering endpoints."""

from __future__ import annotations

import logging
import re
import time

from fastapi import APIRouter, HTTPException, Request

from backend.generation.answering import wants_comparison, wants_summary
from backend.generation.llm import LLMError
from backend.generation.answering import NOT_FOUND_MESSAGE
from backend.models import AskRequest, AskResponse, LLMStatusOut, SourceOut
from backend.services.library import LibraryService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["ask"])


def get_library(request: Request) -> LibraryService:
    return request.app.state.library


@router.get("/llm/status", response_model=LLMStatusOut)
def llm_status(request: Request) -> LLMStatusOut:
    """So the UI can say "Ollama isn't running" instead of failing on ask."""
    library = get_library(request)
    available, detail = library.llm.available()
    return LLMStatusOut(
        available=available,
        provider=library.config.llm.provider,
        model=library.llm.model,
        detail=detail,
    )


def _summary_as_answer(
    library: LibraryService, payload: AskRequest, question: str, started: float
) -> AskResponse:
    """Render a structured summary through the answer shape the UI already knows."""
    candidates = payload.document_ids or [
        d.id for d in library.list_documents() if d.status == "ready"
    ]
    if len(candidates) != 1:
        # Deliberately not a silent guess. Summarising five papers into one
        # blob would be slow and would answer a question nobody asked.
        return AskResponse(
            question=question,
            answer=(
                f"Select a single paper to summarise — {len(candidates)} are currently "
                f"in scope. Tick one paper above and ask again."
            ),
            found=False, sources=[], cited_indexes=[],
            model=library.llm.model,
            took_ms=int((time.perf_counter() - started) * 1000),
        )

    summary = library.summarize(candidates[0])
    if summary is None:
        raise HTTPException(status_code=404, detail="Document not found")

    sections = [f for f in summary["fields"] if f["value"].strip().lower() != "not reported"]
    if not sections:
        return AskResponse(
            question=question, answer=NOT_FOUND_MESSAGE, found=False,
            sources=[], cited_indexes=[], model=summary.get("model", ""),
            took_ms=int((time.perf_counter() - started) * 1000),
        )

    body = "\n\n".join(
        f"**{f['label']}**\n{f['value']} "
        + "".join(f"[{c}]" for c in f["citations"])
        for f in sections
    )
    cited = sorted({c for f in sections for c in f["citations"]})

    return AskResponse(
        question=question,
        answer=f"**Summary of {summary.get('title') or summary['filename']}**\n\n{body}",
        found=True,
        sources=[SourceOut(**s) for s in summary["sources"]],
        cited_indexes=cited,
        model=summary.get("model", ""),
        took_ms=int((time.perf_counter() - started) * 1000),
    )


# Rows worth putting side by side. Fifteen would not be readable, and the ones
# left out (loss function, authors, year) rarely differentiate two papers.
COMPARISON_DIMENSIONS = (
    "task", "dataset", "dataset_size", "model_architecture",
    "method", "training_procedure", "evaluation_metrics", "main_results",
    "limitations",
)

_NUMBER_RE = re.compile(r"\d[\d,.]*")

SYNTHESIS_PROMPT = """You compare scientific papers.

You will be given a table of facts already extracted from several papers.
Write two to four sentences describing how the papers relate: what they have in
common and where they diverge.

Rules:
- Use ONLY the values in the table. Never add facts, numbers or claims.
- Refer to papers by the short names given in the table header.
- If the papers address unrelated problems, say so plainly.
- Plain prose. No markdown, no bullet points, no citation markers."""


def _short_name(record) -> str:
    name = record.title or record.filename
    return name[:48] + ("…" if len(name) > 48 else "")


def _comparison_as_answer(
    library: LibraryService, document_ids: list[str], question: str, started: float
) -> AskResponse:
    """Compare papers against each other, dimension by dimension.

    Built from each paper's cached structured extraction rather than from raw
    retrieved passages. That is the whole point: retrieving passages for the
    word "comparison" finds the comparisons each paper makes *internally*
    (CNN vs Random Forest), not the comparison between the papers.
    """
    records, extractions = [], []
    for document_id in document_ids:
        record = library.get_document(document_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Unknown document {document_id}")
        extraction = library.extract_details(document_id)
        if extraction is None:
            continue
        records.append(record)
        extractions.append(extraction)

    if len(records) < 2:
        return AskResponse(
            question=question,
            answer="Select at least two indexed papers to compare.",
            found=False, sources=[], cited_indexes=[],
            model=library.llm.model,
            took_ms=int((time.perf_counter() - started) * 1000),
        )

    # Merge every paper's evidence into one numbered list, remapping each
    # paper's local citation numbers into the shared numbering.
    sources: list[dict] = []
    field_maps: list[dict[str, dict]] = []
    for extraction in extractions:
        remap: dict[int, int] = {}
        for source in extraction["sources"]:
            merged = dict(source)
            merged["index"] = len(sources) + 1
            remap[source["index"]] = merged["index"]
            sources.append(merged)
        field_maps.append({
            f["name"]: {
                "value": f["value"],
                "citations": [remap[c] for c in f["citations"] if c in remap],
            }
            for f in extraction["fields"]
        })

    labels = {f["name"]: f["label"] for f in extractions[0]["fields"]}
    names = [_short_name(r) for r in records]

    def reported(value: str) -> bool:
        return value.strip().lower() not in {"not reported", "", "n/a", "none", "unknown"}

    rows = []
    for dimension in COMPARISON_DIMENSIONS:
        cells = [fm.get(dimension, {"value": "Not reported", "citations": []})
                 for fm in field_maps]
        rows.append({
            "label": labels.get(dimension, dimension),
            "cells": cells,
            # A dimension only compares anything if at least two papers report it.
            "comparable": sum(1 for c in cells if reported(c["value"])) >= 2,
        })

    comparable = [r for r in rows if r["comparable"]]

    # Honest failure: papers with no dimension reported by two of them have no
    # common ground, and inventing one would be worse than saying so.
    if not comparable:
        listed = "\n".join(f"- {n}" for n in names)
        return AskResponse(
            question=question,
            answer=(
                f"These papers cannot be meaningfully compared on the dimensions "
                f"LocalScholar extracts.\n\n{listed}\n\nNo dimension (dataset, "
                f"method, architecture, metrics, results, limitations) is reported by "
                f"at least two of them — they address different kinds of problem, or "
                f"the details are not stated in terms these papers share. "
                f"Open **Paper details** for each one to see what was found individually."
            ),
            found=False,
            sources=[SourceOut(**s) for s in sources],
            cited_indexes=[],
            model=extractions[0].get("model", ""),
            took_ms=int((time.perf_counter() - started) * 1000),
        )

    header = " | ".join(["Dimension"] + names)
    divider = " | ".join(["---"] * (len(names) + 1))
    body_rows = []
    for row in rows:
        cells = [
            (c["value"].replace("|", "/").replace("\n", " ")
             + "".join(f"[{i}]" for i in c["citations"]))
            for c in row["cells"]
        ]
        body_rows.append(" | ".join([row["label"]] + cells))
    table = "\n".join([header, divider, *body_rows])

    plain_table = "\n".join(
        f"{row['label']}: " + "; ".join(
            f"{name} = {c['value']}" for name, c in zip(names, row["cells"])
        )
        for row in comparable
    )

    synthesis = ""
    try:
        candidate = library.llm.complete(
            SYNTHESIS_PROMPT, f"Table:\n{plain_table}\n\nHow do these papers relate?"
        ).strip()
        # The synthesis restates the table, so any number it contains must
        # already be in the table. A number that is not is invented.
        table_numbers = set(_NUMBER_RE.findall(plain_table))
        if all(n in table_numbers for n in _NUMBER_RE.findall(candidate)):
            synthesis = candidate
        else:
            logger.warning("Comparison synthesis introduced unsupported numbers; dropped.")
    except LLMError as exc:
        logger.warning("Comparison synthesis unavailable: %s", exc)

    skipped = [r["label"] for r in rows if not r["comparable"]]
    notes = [f"Compared on {len(comparable)} of {len(rows)} dimensions."]
    if skipped:
        notes.append(f"Not reported by at least two papers: {', '.join(skipped)}.")

    answer = f"**Comparing {len(records)} papers**\n\n"
    if synthesis:
        answer += synthesis + "\n\n"
    answer += table + "\n\n_" + " ".join(notes) + "_"

    cited = sorted({c for r in rows for cell in r["cells"] for c in cell["citations"]})
    return AskResponse(
        question=question, answer=answer, found=True,
        sources=[SourceOut(**s) for s in sources],
        cited_indexes=cited,
        model=extractions[0].get("model", ""),
        took_ms=int((time.perf_counter() - started) * 1000),
    )


@router.post("/ask", response_model=AskResponse)
def ask(request: Request, payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    library = get_library(request)
    started = time.perf_counter()

    # "Summarise this paper" is a different job from "what dataset did they
    # use?", and answering it through the question pipeline produced a
    # confident "I could not find sufficient evidence" for papers that were
    # sitting right there. Route it to the summary pipeline instead.
    scope = payload.document_ids or [
        d.id for d in library.list_documents() if d.status == "ready"
    ]

    if wants_summary(question):
        return _summary_as_answer(library, payload, question, started)

    # Comparing papers only makes sense with more than one in scope. With a
    # single paper selected, "what does it compare?" is an ordinary question
    # about that paper's own experiments, so it falls through below.
    if wants_comparison(question) and len(scope) > 1:
        return _comparison_as_answer(library, scope, question, started)

    chunks = library.engine.hybrid_search(
        question,
        top_k=payload.top_k,
        document_ids=payload.document_ids or None,
    )

    try:
        result = library.answers.answer(question, chunks)
    except LLMError as exc:
        # 503, not 500: the app is fine, the model server is not, and the
        # message tells the user exactly which.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return AskResponse(
        question=question,
        answer=result.answer,
        found=result.found,
        sources=[SourceOut.from_source(s) for s in result.sources],
        cited_indexes=result.cited_indexes,
        model=result.model,
        took_ms=int((time.perf_counter() - started) * 1000),
    )
