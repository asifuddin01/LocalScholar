"""SQLite catalogue of the paper library.

SQLite rather than a "real" database because the entire point of this project
is that it runs on one laptop with no services to start. It holds the library
metadata and the extracted page text; embeddings live in the vector store.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

# Document lifecycle. The UI polls until a document leaves PROCESSING.
STATUS_PROCESSING = "processing"
STATUS_READY = "ready"
STATUS_FAILED = "failed"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    sha256       TEXT NOT NULL UNIQUE,
    filename     TEXT NOT NULL,
    title        TEXT,
    page_count   INTEGER NOT NULL DEFAULT 0,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    sections     TEXT NOT NULL DEFAULT '[]',
    status       TEXT NOT NULL,
    error        TEXT,
    size_bytes   INTEGER NOT NULL DEFAULT 0,
    stored_path  TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    document_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number  INTEGER NOT NULL,
    text         TEXT NOT NULL,
    PRIMARY KEY (document_id, page_number)
);
"""


@dataclass
class DocumentRecord:
    id: str
    sha256: str
    filename: str
    title: str | None
    page_count: int
    chunk_count: int
    sections: list[str]
    status: str
    error: str | None
    size_bytes: int
    stored_path: str
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DocumentRecord":
        return cls(
            id=row["id"],
            sha256=row["sha256"],
            filename=row["filename"],
            title=row["title"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            sections=json.loads(row["sections"]),
            status=row["status"],
            error=row["error"],
            size_bytes=row["size_bytes"],
            stored_path=row["stored_path"],
            created_at=row["created_at"],
        )


class Catalogue:
    """Thin data-access layer. One connection per operation, guarded by a lock.

    FastAPI runs sync endpoints in a threadpool, so connections must not be
    shared across threads. Opening per call is cheap for a local SQLite file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- reads --------------------------------------------------------------

    def list_documents(self) -> list[DocumentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [DocumentRecord.from_row(r) for r in rows]

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return DocumentRecord.from_row(row) if row else None

    def find_by_hash(self, sha256: str) -> DocumentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return DocumentRecord.from_row(row) if row else None

    def get_pages(self, document_id: str) -> list[tuple[int, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT page_number, text FROM pages WHERE document_id = ? "
                "ORDER BY page_number",
                (document_id,),
            ).fetchall()
        return [(r["page_number"], r["text"]) for r in rows]

    # --- writes -------------------------------------------------------------

    def create_document(
        self, *, document_id: str, sha256: str, filename: str,
        size_bytes: int, stored_path: Path,
    ) -> DocumentRecord:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO documents (id, sha256, filename, status, size_bytes, "
                "stored_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (document_id, sha256, filename, STATUS_PROCESSING, size_bytes,
                 str(stored_path), created_at),
            )
        record = self.get_document(document_id)
        assert record is not None
        return record

    def mark_ready(
        self, document_id: str, *, title: str | None, page_count: int,
        sections: list[str], chunk_count: int = 0,
    ) -> None:
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status = ?, title = ?, page_count = ?, "
                "sections = ?, chunk_count = ?, error = NULL WHERE id = ?",
                (STATUS_READY, title, page_count, json.dumps(sections),
                 chunk_count, document_id),
            )

    def mark_failed(self, document_id: str, error: str) -> None:
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status = ?, error = ? WHERE id = ?",
                (STATUS_FAILED, error, document_id),
            )

    def replace_pages(self, document_id: str, pages: list[tuple[int, str]]) -> None:
        with self._write_lock, self._connect() as conn:
            conn.execute("DELETE FROM pages WHERE document_id = ?", (document_id,))
            conn.executemany(
                "INSERT INTO pages (document_id, page_number, text) VALUES (?, ?, ?)",
                [(document_id, n, t) for n, t in pages],
            )

    def delete_document(self, document_id: str) -> DocumentRecord | None:
        record = self.get_document(document_id)
        if record is None:
            return None
        with self._write_lock, self._connect() as conn:
            conn.execute("DELETE FROM pages WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return record
