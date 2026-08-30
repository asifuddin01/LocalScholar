"""Grounded question answering endpoints."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request

from backend.generation.llm import LLMError
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


@router.post("/ask", response_model=AskResponse)
def ask(request: Request, payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    library = get_library(request)
    started = time.perf_counter()

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
