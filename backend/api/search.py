"""Retrieval endpoints.

Dense and lexical retrieval are exposed as separate methods rather than hidden
behind one "search" call. Two reasons: it makes the difference between them
visible while developing, and the retrieval ablation (milestone 6) needs to
measure each one independently.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request

from backend.models import SearchRequest, SearchResponse, SearchResultOut
from backend.services.library import LibraryService

router = APIRouter(prefix="/api", tags=["search"])


def get_library(request: Request) -> LibraryService:
    return request.app.state.library


@router.post("/search", response_model=SearchResponse)
def search(request: Request, payload: SearchRequest) -> SearchResponse:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    library = get_library(request)
    engine = library.engine
    document_ids = payload.document_ids or None

    # dense_top_k / bm25_top_k size the *candidate pool* that fusion and
    # reranking consume. A caller asking this endpoint directly wants the
    # final, presentable number of results, so that is the default here.
    top_k = payload.top_k or library.config.retrieval.final_top_k

    started = time.perf_counter()
    if payload.method == "bm25":
        results = engine.bm25_search(query, top_k=top_k, document_ids=document_ids)
    elif payload.method == "dense":
        results = engine.dense_search(query, top_k=top_k, document_ids=document_ids)
    else:
        results = engine.hybrid_search(query, top_k=top_k, document_ids=document_ids)

    return SearchResponse(
        query=query,
        method=payload.method,
        results=[SearchResultOut.from_scored(r) for r in results],
        took_ms=int((time.perf_counter() - started) * 1000),
    )
