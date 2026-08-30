"""End-to-end search over the API: upload a paper, then find things in it."""

from __future__ import annotations

import os

import pytest

from tests.conftest import build_paper_pdf


def upload(client, name, data):
    return client.post(
        "/api/documents", files=[("files", (name, data, "application/pdf"))]
    )


@pytest.fixture
def indexed(client, paper_pdf_bytes):
    upload(client, "paper.pdf", paper_pdf_bytes)
    return client, client.get("/api/documents").json()[0]


def search(client, query, **kwargs):
    return client.post("/api/search", json={"query": query, **kwargs}).json()


def test_upload_produces_chunks(indexed):
    _, document = indexed
    assert document["status"] == "ready"
    assert document["chunk_count"] > 0


def test_bm25_finds_an_exact_term_with_the_right_page(indexed):
    client, _ = indexed
    body = search(client, "KiTS19", method="bm25")
    assert body["results"]
    top = body["results"][0]
    assert "KiTS19" in top["text"]
    # The methods paragraph is on page 2 of the generated paper.
    assert top["page_number"] == 2
    assert top["section"] == "2 Methods"


def test_dense_search_returns_results(indexed):
    client, _ = indexed
    body = search(client, "which dataset was used for training", method="dense")
    assert body["results"]
    assert all(r["score"] > 0 for r in body["results"])


def test_results_carry_everything_a_citation_needs(indexed):
    client, _ = indexed
    result = search(client, "KiTS19", method="bm25")["results"][0]
    for field in ("chunk_id", "document_id", "filename", "page_number", "text"):
        assert result[field], f"missing {field}"


def test_search_can_be_restricted_to_selected_papers(client):
    upload(client, "a.pdf", build_paper_pdf("Paper A"))
    upload(client, "b.pdf", build_paper_pdf("Paper B"))
    documents = client.get("/api/documents").json()
    chosen = documents[0]["id"]

    body = search(client, "dataset", method="bm25", document_ids=[chosen])
    assert body["results"]
    assert {r["document_id"] for r in body["results"]} == {chosen}


def test_search_across_the_whole_library_by_default(client):
    upload(client, "a.pdf", build_paper_pdf("Paper A"))
    upload(client, "b.pdf", build_paper_pdf("Paper B"))
    body = search(client, "KiTS19 dataset", method="bm25")
    assert len({r["document_id"] for r in body["results"]}) == 2


def test_top_k_is_respected(indexed):
    client, _ = indexed
    assert len(search(client, "the", method="dense", top_k=2)["results"]) <= 2


def test_empty_query_is_rejected(client):
    assert client.post("/api/search", json={"query": "   "}).status_code == 400


def test_search_on_an_empty_library_returns_no_results(client):
    body = search(client, "anything", method="bm25")
    assert body["results"] == []


def test_deleting_a_paper_removes_it_from_search(indexed):
    client, document = indexed
    assert search(client, "KiTS19", method="bm25")["results"]

    client.delete(f"/api/documents/{document['id']}")

    assert search(client, "KiTS19", method="bm25")["results"] == []
    assert search(client, "KiTS19", method="dense")["results"] == []


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("LOCALSCHOLAR_SKIP_MODEL_TEST") == "1",
    reason="explicitly disabled",
)
def test_real_embedding_model_ranks_a_paraphrase_above_an_unrelated_chunk(tmp_path, monkeypatch):
    """The one test that exercises the actual model.

    The hashing stub used elsewhere is lexical, so it cannot show that dense
    retrieval does the thing dense retrieval is *for*: matching a question to
    a passage that shares no words with it. Downloads bge-small on first run,
    which is why it is opt-in (`pytest -m slow`).
    """
    from fastapi.testclient import TestClient

    monkeypatch.setenv("LOCALSCHOLAR_DATA_DIR", str(tmp_path / "data"))
    from backend.main import app

    with TestClient(app) as real_client:
        upload(real_client, "paper.pdf", build_paper_pdf())
        # No word here appears in the methods paragraph except "training".
        body = search(real_client, "how many scans were annotated?", method="dense")
        assert body["results"]
        assert "KiTS19" in body["results"][0]["text"]


@pytest.mark.slow
def test_large_document_indexes_from_a_background_thread(tmp_path, monkeypatch):
    """Regression test for a silent deadlock.

    Indexing runs in a background thread so that uploads stay responsive.
    onnxruntime's default batch size of 256 deadlocks inside `session.run()`
    when called off the main thread on macOS/ARM: no exception, no timeout,
    the document just sits at "processing" forever. Small fixtures never hit
    it, so this uses a document big enough to fill more than one batch.
    """
    from fastapi.testclient import TestClient

    from tests.conftest import build_large_paper_pdf

    monkeypatch.setenv("LOCALSCHOLAR_DATA_DIR", str(tmp_path / "data"))
    from backend.main import app

    with TestClient(app) as real_client:
        upload(real_client, "long.pdf", build_large_paper_pdf(pages=70))
        document = real_client.get("/api/documents").json()[0]

        assert document["status"] == "ready", document.get("error")
        # Must exceed onnxruntime's default batch of 256, or the fixture would
        # pass even with the deadlocking configuration restored.
        assert document["chunk_count"] > 260, "fixture too small to hit the deadlock"
        assert search(real_client, "cross validation", method="dense")["results"]
