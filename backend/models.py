"""Pydantic schemas for the HTTP boundary.

Kept separate from the internal dataclasses on purpose: the API contract and
the parser's data model change for different reasons.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from backend.db import DocumentRecord
from backend.generation.answering import Source
from backend.retrieval.vector_store import ScoredChunk


class DocumentOut(BaseModel):
    id: str
    filename: str
    title: str | None = None
    page_count: int = 0
    chunk_count: int = 0
    sections: list[str] = Field(default_factory=list)
    status: str
    error: str | None = None
    size_bytes: int = 0
    created_at: str

    @classmethod
    def from_record(cls, record: DocumentRecord) -> "DocumentOut":
        return cls(
            id=record.id,
            filename=record.filename,
            title=record.title,
            page_count=record.page_count,
            chunk_count=record.chunk_count,
            sections=record.sections,
            status=record.status,
            error=record.error,
            size_bytes=record.size_bytes,
            created_at=record.created_at,
        )


class UploadResultOut(BaseModel):
    document: DocumentOut
    duplicate: bool = False
    message: str | None = None


class UploadResponse(BaseModel):
    results: list[UploadResultOut] = Field(default_factory=list)
    rejected: list[dict[str, str]] = Field(default_factory=list)


class PageOut(BaseModel):
    page_number: int
    text: str


class DocumentDetailOut(BaseModel):
    document: DocumentOut
    pages: list[PageOut] = Field(default_factory=list)


# --- search -----------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str
    # Empty means "search the whole library". Selecting papers filters inside
    # the search rather than after it, so a filtered search still returns a
    # full top_k from the chosen papers.
    document_ids: list[str] = Field(default_factory=list)
    method: Literal["dense", "bm25", "hybrid"] = "hybrid"
    top_k: int | None = None


class SearchResultOut(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    title: str | None = None
    page_number: int
    section: str | None = None
    text: str
    score: float

    @classmethod
    def from_scored(cls, scored: "ScoredChunk") -> "SearchResultOut":
        return cls(
            chunk_id=scored.chunk_id,
            document_id=scored.document_id,
            filename=scored.filename,
            title=scored.title,
            page_number=scored.page_number,
            section=scored.section,
            text=scored.text,
            score=round(scored.score, 6),
        )


class SearchResponse(BaseModel):
    query: str
    method: str
    results: list[SearchResultOut] = Field(default_factory=list)
    took_ms: int = 0


# --- question answering -----------------------------------------------------


class AskRequest(BaseModel):
    question: str
    document_ids: list[str] = Field(default_factory=list)
    top_k: int | None = None


class SourceOut(BaseModel):
    index: int
    chunk_id: str
    document_id: str
    filename: str
    title: str | None = None
    page_number: int
    section: str | None = None
    text: str
    score: float

    @classmethod
    def from_source(cls, source: "Source") -> "SourceOut":
        return cls(
            index=source.index,
            chunk_id=source.chunk_id,
            document_id=source.document_id,
            filename=source.filename,
            title=source.title,
            page_number=source.page_number,
            section=source.section,
            text=source.text,
            score=round(source.score, 6),
        )


class AskResponse(BaseModel):
    question: str
    answer: str
    # False means the papers did not support an answer. The UI must not present
    # a not-found result as if it were a finding.
    found: bool
    sources: list[SourceOut] = Field(default_factory=list)
    cited_indexes: list[int] = Field(default_factory=list)
    model: str = ""
    took_ms: int = 0


class LLMStatusOut(BaseModel):
    available: bool
    provider: str
    model: str
    detail: str = ""
