"""Library service: turns an uploaded file into an indexed document.

The upload endpoint stays fast by doing only the cheap, must-not-fail work
(validate, hash, write to disk, create the row). Parsing runs in a background
task and flips the document's status, which is why the UI polls.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from backend.config import Config
from backend.db import STATUS_READY, Catalogue, DocumentRecord
from backend.ingestion.chunking import chunk_document
from backend.ingestion.pdf_parser import PDFParseError, parse_pdf
from backend.generation.answering import AnswerService
from backend.generation.extraction import (
    SUMMARY_GROUPS,
    SUMMARY_SYSTEM_PROMPT,
    ExtractionService,
)
from backend.generation.llm import create_llm_provider
from backend.retrieval.engine import RetrievalEngine

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF"


class UploadRejected(Exception):
    """The uploaded bytes are not something we can index."""


class LibraryService:
    def __init__(self, config: Config) -> None:
        self.config = config
        config.ensure_directories()
        self.catalogue = Catalogue(config.storage.catalogue_path)
        self.engine = RetrievalEngine(config, self.catalogue)
        self.llm = create_llm_provider(config.llm)
        self.answers = AnswerService(self.llm, config.llm.context_char_budget)
        self.extractions = ExtractionService(
            self.engine, self.llm, config.llm.context_char_budget
        )

    def start(self) -> list[str]:
        """Restore in-memory state at startup.

        The BM25 index lives in memory, so it is rebuilt from SQLite on every
        boot. Any document marked ready but holding no chunks was interrupted
        mid-index (a crash, or a library created before indexing existed), so
        it is queued to be processed again rather than sitting in the library
        looking searchable while matching nothing.
        """
        self.engine.rebuild_bm25()

        # A damaged vector index is silently fatal: searches either fail with a
        # numpy broadcast error or quietly return nothing. Checked every boot,
        # and repaired from SQLite, which always has the text.
        if self.engine.needs_repair():
            try:
                self.engine.repair()
            except Exception:  # noqa: BLE001 - never block startup on this
                logger.exception("Vector index repair failed")

        stale = [
            record.id
            for record in self.catalogue.list_documents()
            if record.status == STATUS_READY and record.chunk_count == 0
        ]
        if stale:
            logger.info("Re-indexing %d document(s) with no chunks", len(stale))
        return stale

    # --- upload -------------------------------------------------------------

    def ingest_upload(self, filename: str, data: bytes) -> tuple[DocumentRecord, bool]:
        """Store an uploaded PDF and register it.

        Returns (record, was_already_present). Content is addressed by SHA-256,
        so re-uploading the same paper -- even under a different filename --
        returns the existing document instead of indexing it twice.
        """
        display_name = Path(filename or "upload.pdf").name or "upload.pdf"

        if not data:
            raise UploadRejected(f"{display_name} is empty.")
        if len(data) > self.config.ingestion.max_upload_bytes:
            raise UploadRejected(
                f"{display_name} is {len(data) / 1024 / 1024:.1f}MB, over the "
                f"{self.config.ingestion.max_upload_mb}MB limit."
            )
        if not data.lstrip()[:4] == PDF_MAGIC:
            raise UploadRejected(f"{display_name} does not look like a PDF file.")

        sha256 = hashlib.sha256(data).hexdigest()
        existing = self.catalogue.find_by_hash(sha256)
        if existing is not None:
            logger.info("Skipping duplicate upload %s (matches %s)",
                        display_name, existing.id)
            return existing, True

        document_id = sha256[:16]
        # Stored under the content id, never the user-supplied name: an upload
        # can't then escape the uploads directory or collide with another file.
        stored_path = self.config.storage.uploads_dir / f"{document_id}.pdf"
        stored_path.write_bytes(data)

        record = self.catalogue.create_document(
            document_id=document_id,
            sha256=sha256,
            filename=display_name,
            size_bytes=len(data),
            stored_path=stored_path,
        )
        return record, False

    # --- processing ---------------------------------------------------------

    def process_document(self, document_id: str) -> None:
        """Parse a stored PDF and record its pages. Runs in the background."""
        record = self.catalogue.get_document(document_id)
        if record is None:
            logger.warning("process_document called for unknown id %s", document_id)
            return
        try:
            parsed = parse_pdf(record.stored_path)
            self.catalogue.replace_pages(
                document_id, [(p.page_number, p.text) for p in parsed.pages]
            )

            chunks = chunk_document(
                document_id=document_id,
                filename=record.filename,
                parsed=parsed,
                config=self.config.chunking,
            )
            # SQLite first: it is the durable copy the BM25 index is rebuilt
            # from, so it must be written before anything depends on it.
            self.catalogue.replace_chunks(document_id, chunks)
            self.engine.index_chunks(chunks)
            self.engine.rebuild_bm25()

            self.catalogue.mark_ready(
                document_id,
                title=parsed.title,
                page_count=parsed.page_count,
                sections=parsed.sections,
                chunk_count=len(chunks),
            )
            logger.info(
                "Indexed %s (%d pages, %d chunks)",
                record.filename, parsed.page_count, len(chunks),
            )
        except PDFParseError as exc:
            self.catalogue.mark_failed(document_id, str(exc))
            logger.warning("Failed to parse %s: %s", record.filename, exc)
        except Exception as exc:  # noqa: BLE001 - a bad PDF must not kill the server
            self.catalogue.mark_failed(document_id, f"Unexpected error: {exc}")
            logger.exception("Unexpected error processing %s", record.filename)

    # --- reads / deletes ----------------------------------------------------

    def list_documents(self) -> list[DocumentRecord]:
        return self.catalogue.list_documents()

    def get_document(self, document_id: str) -> DocumentRecord | None:
        return self.catalogue.get_document(document_id)

    def get_pages(self, document_id: str) -> list[tuple[int, str]]:
        return self.catalogue.get_pages(document_id)

    def delete_document(self, document_id: str) -> bool:
        record = self.catalogue.delete_document(document_id)
        if record is None:
            return False
        Path(record.stored_path).unlink(missing_ok=True)
        self.engine.remove_document(document_id)
        return True

    def extract_details(self, document_id: str, refresh: bool = False) -> dict | None:
        """Structured details for one paper, cached after the first run."""
        record = self.catalogue.get_document(document_id)
        if record is None:
            return None
        if not refresh:
            cached = self.catalogue.get_extraction(document_id)
            if cached is not None:
                return cached

        extraction = self.extractions.extract(document_id, record.filename)
        payload = extraction.to_dict()
        # The title is known from parsing page one; it is prepended as a field
        # so the details view and the comparison table have a complete row set
        # without the model ever being asked to guess it.
        payload["title"] = record.title
        payload["fields"].insert(0, {
            "name": "title", "label": "Title",
            "value": record.title or record.filename, "citations": [],
        })
        self.catalogue.save_extraction(document_id, payload.get("model", ""), payload)
        return payload

    def summarize(self, document_id: str, refresh: bool = False) -> dict | None:
        """A structured, cited summary of one paper.

        Runs the same schema-driven pipeline as the details view, with the
        summary schema and prose-level citation verification. Cached, because
        it costs three model calls.
        """
        record = self.catalogue.get_document(document_id)
        if record is None:
            return None
        if not refresh:
            cached = self.catalogue.get_summary(document_id)
            if cached is not None:
                return cached

        summary = self.extractions.extract(
            document_id, record.filename,
            groups=SUMMARY_GROUPS, system_prompt=SUMMARY_SYSTEM_PROMPT, prose=True,
        )
        payload = summary.to_dict()
        payload["title"] = record.title or record.filename
        self.catalogue.save_summary(document_id, payload.get("model", ""), payload)
        return payload

    def close(self) -> None:
        self.engine.close()
