"""Paper details and comparison.

Both are views over the same structured extraction, so a paper can never
describe its dataset one way in the details panel and another way in a
comparison table.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from backend.generation.extraction import ALL_FIELDS, NOT_REPORTED
from backend.services.library import LibraryService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["research"])

# The rows worth putting side by side. The full field list is available in the
# details view; a comparison table with fifteen rows is not readable.
COMPARISON_FIELDS = (
    "dataset", "dataset_size", "task", "model_architecture",
    "training_procedure", "evaluation_metrics", "main_results", "limitations",
)


def get_library(request: Request) -> LibraryService:
    return request.app.state.library


@router.get("/documents/{document_id}/details")
def paper_details(request: Request, document_id: str, refresh: bool = False) -> dict:
    library = get_library(request)
    available, detail = library.llm.available()
    if not available:
        raise HTTPException(status_code=503, detail=detail)

    payload = library.extract_details(document_id, refresh=refresh)
    if payload is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return payload


@router.get("/documents/{document_id}/summary")
def paper_summary(request: Request, document_id: str, refresh: bool = False) -> dict:
    library = get_library(request)
    available, detail = library.llm.available()
    if not available:
        raise HTTPException(status_code=503, detail=detail)

    payload = library.summarize(document_id, refresh=refresh)
    if payload is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return payload


@router.post("/compare")
def compare(request: Request, payload: dict) -> dict:
    document_ids = payload.get("document_ids") or []
    if len(document_ids) < 2:
        raise HTTPException(status_code=400, detail="Select at least two papers to compare.")

    library = get_library(request)
    available, detail = library.llm.available()
    if not available:
        raise HTTPException(status_code=503, detail=detail)

    columns, extractions = [], []
    for document_id in document_ids:
        record = library.get_document(document_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Unknown document {document_id}")
        extraction = library.extract_details(document_id)
        columns.append({
            "document_id": document_id,
            "filename": record.filename,
            "title": record.title,
        })
        extractions.append({f["name"]: f for f in extraction["fields"]})

    labels = dict(ALL_FIELDS)
    rows = []
    for name in COMPARISON_FIELDS:
        cells = []
        for extraction in extractions:
            entry = extraction.get(name)
            cells.append({
                "value": (entry or {}).get("value", NOT_REPORTED),
                "citations": (entry or {}).get("citations", []),
            })
        rows.append({"field": name, "label": labels.get(name, name), "cells": cells})

    return {"columns": columns, "rows": rows}
