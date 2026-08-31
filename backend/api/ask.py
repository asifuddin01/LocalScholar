"""Grounded question answering endpoints."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request

from backend.generation.answering import wants_summary
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
    if wants_summary(question):
        return _summary_as_answer(library, payload, question, started)

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
