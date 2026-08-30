"""API-level tests for the document library.

These drive the real FastAPI app against a temporary data directory, so they
cover the parts unit tests miss: multipart upload, background processing, and
the status transitions the UI polls on.
"""

from __future__ import annotations

from tests.conftest import build_blank_pdf, build_paper_pdf


def upload(client, name, data):
    return client.post(
        "/api/documents", files=[("files", (name, data, "application/pdf"))]
    )


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_library_starts_empty(client):
    assert client.get("/api/documents").json() == []


def test_upload_indexes_the_paper(client, paper_pdf_bytes):
    response = upload(client, "paper.pdf", paper_pdf_bytes)
    assert response.status_code == 200

    body = response.json()
    assert body["rejected"] == []
    assert len(body["results"]) == 1
    assert body["results"][0]["duplicate"] is False

    # Background processing runs before the test client returns.
    document = client.get("/api/documents").json()[0]
    assert document["status"] == "ready"
    assert document["page_count"] == 2
    assert document["filename"] == "paper.pdf"
    assert document["title"] == "A Study of Local Retrieval"
    assert "2 Methods" in document["sections"]
    assert document["error"] is None


def test_reuploading_the_same_content_does_not_index_it_twice(client, paper_pdf_bytes):
    upload(client, "paper.pdf", paper_pdf_bytes)
    # Same bytes, different filename: content addressing should still catch it.
    response = upload(client, "paper-copy.pdf", paper_pdf_bytes)

    assert response.json()["results"][0]["duplicate"] is True
    assert len(client.get("/api/documents").json()) == 1


def test_different_papers_are_kept_separate(client):
    upload(client, "a.pdf", build_paper_pdf("Paper A"))
    upload(client, "b.pdf", build_paper_pdf("Paper B"))
    titles = {d["title"] for d in client.get("/api/documents").json()}
    assert titles == {"Paper A", "Paper B"}


def test_non_pdf_upload_is_rejected(client):
    response = upload(client, "notes.txt", b"just some text, not a PDF")
    body = response.json()
    assert body["results"] == []
    assert body["rejected"][0]["filename"] == "notes.txt"
    assert "does not look like a PDF" in body["rejected"][0]["reason"]


def test_empty_upload_is_rejected(client):
    assert "empty" in upload(client, "empty.pdf", b"").json()["rejected"][0]["reason"]


def test_one_bad_file_does_not_sink_the_batch(client, paper_pdf_bytes):
    """A ten-paper drop should not be lost because one file is broken."""
    response = client.post(
        "/api/documents",
        files=[
            ("files", ("good.pdf", paper_pdf_bytes, "application/pdf")),
            ("files", ("bad.txt", b"nope", "text/plain")),
        ],
    )
    body = response.json()
    assert len(body["results"]) == 1
    assert len(body["rejected"]) == 1


def test_unparseable_pdf_is_recorded_as_failed_not_ready(client):
    """A scanned PDF must surface an error, not sit there looking indexed."""
    upload(client, "scanned.pdf", build_blank_pdf())
    document = client.get("/api/documents").json()[0]
    assert document["status"] == "failed"
    assert "scanned" in document["error"]


def test_document_detail_returns_page_text(client, paper_pdf_bytes):
    upload(client, "paper.pdf", paper_pdf_bytes)
    document_id = client.get("/api/documents").json()[0]["id"]

    detail = client.get(f"/api/documents/{document_id}").json()
    assert len(detail["pages"]) == 2
    assert detail["pages"][0]["page_number"] == 1
    assert "KiTS19" in detail["pages"][1]["text"]


def test_unknown_document_returns_404(client):
    assert client.get("/api/documents/does-not-exist").status_code == 404


def test_delete_removes_the_document(client, paper_pdf_bytes):
    upload(client, "paper.pdf", paper_pdf_bytes)
    document_id = client.get("/api/documents").json()[0]["id"]

    assert client.delete(f"/api/documents/{document_id}").status_code == 204
    assert client.get("/api/documents").json() == []
    assert client.delete(f"/api/documents/{document_id}").status_code == 404
