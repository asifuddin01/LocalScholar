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
from backend.db import Catalogue, DocumentRecord
from backend.ingestion.pdf_parser import PDFParseError, parse_pdf

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF"


class UploadRejected(Exception):
    """The uploaded bytes are not something we can index."""


class LibraryService:
    def __init__(self, config: Config) -> None:
        self.config = config
        config.ensure_directories()
        self.catalogue = Catalogue(config.storage.catalogue_path)

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
            self.catalogue.mark_ready(
                document_id,
                title=parsed.title,
                page_count=parsed.page_count,
                sections=parsed.sections,
            )
            logger.info("Indexed %s (%d pages)", record.filename, parsed.page_count)
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
        return True
