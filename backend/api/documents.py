"""Document library endpoints."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, UploadFile

from backend.models import (
    DocumentDetailOut,
    DocumentOut,
    PageOut,
    UploadResponse,
    UploadResultOut,
)
from backend.services.library import LibraryService, UploadRejected

router = APIRouter(prefix="/api/documents", tags=["documents"])


def get_library(request: Request) -> LibraryService:
    return request.app.state.library


@router.get("", response_model=list[DocumentOut])
def list_documents(request: Request) -> list[DocumentOut]:
    library = get_library(request)
    return [DocumentOut.from_record(r) for r in library.list_documents()]


@router.post("", response_model=UploadResponse)
async def upload_documents(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile],
) -> UploadResponse:
    """Accept one or more PDFs. Each file succeeds or fails independently.

    One malformed PDF in a batch of ten should not lose the other nine, so
    rejections are reported per file rather than failing the whole request.
    """
    library = get_library(request)
    response = UploadResponse()

    for upload in files:
        filename = upload.filename or "upload.pdf"
        try:
            data = await upload.read()
            record, duplicate = library.ingest_upload(filename, data)
        except UploadRejected as exc:
            response.rejected.append({"filename": filename, "reason": str(exc)})
            continue
        finally:
            await upload.close()

        if not duplicate:
            background_tasks.add_task(library.process_document, record.id)

        response.results.append(
            UploadResultOut(
                document=DocumentOut.from_record(record),
                duplicate=duplicate,
                message="Already in your library." if duplicate else None,
            )
        )

    return response


@router.get("/{document_id}", response_model=DocumentDetailOut)
def get_document(request: Request, document_id: str) -> DocumentDetailOut:
    library = get_library(request)
    record = library.get_document(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    pages = [PageOut(page_number=n, text=t) for n, t in library.get_pages(document_id)]
    return DocumentDetailOut(document=DocumentOut.from_record(record), pages=pages)


@router.delete("/{document_id}", status_code=204)
def delete_document(request: Request, document_id: str) -> None:
    library = get_library(request)
    if not library.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
