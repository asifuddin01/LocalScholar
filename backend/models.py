"""Pydantic schemas for the HTTP boundary.

Kept separate from the internal dataclasses on purpose: the API contract and
the parser's data model change for different reasons.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.db import DocumentRecord


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
