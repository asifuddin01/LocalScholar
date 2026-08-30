"""Shared fixtures.

Tests build their PDFs in memory rather than committing binary fixtures, so
the exact layout under test (page breaks, headings, font sizes) is visible in
the test file itself.
"""

from __future__ import annotations

import hashlib
import re

import pymupdf
import pytest

from backend.retrieval.embeddings import EmbeddingProvider

BODY_FONT_SIZE = 10
HEADING_FONT_SIZE = 11      # deliberately NOT large enough to trigger the
                            # font-size heading rule, so the tests exercise the
                            # section-name and numbering rules instead.
TITLE_FONT_SIZE = 20

ABSTRACT_TEXT = (
    "We present a method for evaluating retrieval quality on research papers. "
    "Our approach combines lexical and dense signals."
)
INTRO_TEXT = (
    "Reading research papers is slow. Prior work focuses on generic question "
    "answering rather than on the structured details that reviewers need."
)
METHOD_TEXT = (
    "We train on the KiTS19 dataset, which contains 300 annotated computed "
    "tomography scans. Images are resampled to a fixed spacing before training."
)
RESULT_TEXT = (
    "The proposed model reaches a Dice score of 0.91 on the held-out split, "
    "outperforming the U-Net baseline by 4.2 points."
)


def build_paper_pdf(title: str = "A Study of Local Retrieval") -> bytes:
    """A miniature two-page 'paper' with a title, headings and body text."""
    doc = pymupdf.open()

    page = doc.new_page()
    page.insert_text((72, 110), title, fontsize=TITLE_FONT_SIZE)
    page.insert_text((72, 220), "Abstract", fontsize=HEADING_FONT_SIZE)
    page.insert_textbox(
        pymupdf.Rect(72, 235, 520, 340), ABSTRACT_TEXT, fontsize=BODY_FONT_SIZE
    )
    # Below the title region, so the numbering rule is allowed to fire.
    page.insert_text((72, 430), "1 Introduction", fontsize=HEADING_FONT_SIZE)
    page.insert_textbox(
        pymupdf.Rect(72, 445, 520, 600), INTRO_TEXT, fontsize=BODY_FONT_SIZE
    )

    page = doc.new_page()
    page.insert_text((72, 110), "2 Methods", fontsize=HEADING_FONT_SIZE)
    page.insert_textbox(
        pymupdf.Rect(72, 125, 520, 260), METHOD_TEXT, fontsize=BODY_FONT_SIZE
    )
    page.insert_text((72, 400), "3 Results", fontsize=HEADING_FONT_SIZE)
    page.insert_textbox(
        pymupdf.Rect(72, 415, 520, 560), RESULT_TEXT, fontsize=BODY_FONT_SIZE
    )

    data = doc.tobytes()
    doc.close()
    return data


def build_blank_pdf(pages: int = 2) -> bytes:
    """A PDF with no selectable text, standing in for a scanned document."""
    doc = pymupdf.open()
    for _ in range(pages):
        doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def paper_pdf_bytes() -> bytes:
    return build_paper_pdf()


@pytest.fixture
def paper_pdf(tmp_path, paper_pdf_bytes) -> str:
    path = tmp_path / "paper.pdf"
    path.write_bytes(paper_pdf_bytes)
    return str(path)


class HashingEmbeddingProvider(EmbeddingProvider):
    """A deterministic stand-in for the real embedding model.

    Loading bge-small for every test turned a 0.8s suite into a 92s one, which
    is long enough that nobody runs it. This hashes tokens into a bag-of-words
    vector instead: it needs no model, no download and no network, and it is
    similarity-preserving enough that retrieval plumbing (filtering, ranking,
    top_k, citation mapping) is genuinely exercised.

    It is NOT a semantic model, so it cannot verify retrieval *quality*. That
    is what the opt-in real-model test and the milestone-6 benchmark are for.
    """

    DIMENSION = 64

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    @property
    def model_name(self) -> str:
        return "test-hashing"

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.DIMENSION
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.md5(token.encode()).digest()
            vector[int.from_bytes(digest[:4], "little") % self.DIMENSION] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector] if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient with a temp data directory and a stubbed embedding model."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("LOCALSCHOLAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(
        "backend.retrieval.engine.create_embedding_provider",
        lambda **_: HashingEmbeddingProvider(),
    )
    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client


def build_large_paper_pdf(pages: int = 70) -> bytes:
    """A paper long enough to produce several hundred chunks.

    Used by the background-thread indexing regression test. The onnxruntime
    deadlock only appears once a single batch is large, so this has to produce
    more than 256 chunks -- onnxruntime's default batch size -- or it proves
    nothing. A two-page fixture cannot reproduce it.
    """
    paragraph = (
        "The proposed segmentation network is trained on annotated volumes "
        "using a composite objective. We resample every scan to an isotropic "
        "spacing and normalise intensities per patient before augmentation. "
        "Validation is performed with five-fold cross validation across sites. "
    )
    doc = pymupdf.open()
    for index in range(pages):
        page = doc.new_page()
        page.insert_text((72, 100), f"{index + 1} Experiments", fontsize=HEADING_FONT_SIZE)
        # insert_textbox writes nothing at all when the text overflows the
        # rectangle, so fill it by backing off until it fits rather than
        # guessing a repeat count that happens to work at this font size.
        box = pymupdf.Rect(72, 120, 520, 720)
        for repeat in range(16, 0, -1):
            if page.insert_textbox(box, paragraph * repeat, fontsize=BODY_FONT_SIZE) >= 0:
                break
    data = doc.tobytes()
    doc.close()
    return data


def build_paper_with_running_header(
    pages: int = 6,
    header: str = "Title Suppressed Due to Excessive Length",
    body: str = "The dataset contains 300 annotated volumes from two clinical sites. ",
) -> bytes:
    """Pages stamped with a journal-style running header, numbered per page."""
    doc = pymupdf.open()
    for index in range(pages):
        page = doc.new_page()
        # y=93 on an 842pt page: where LaTeX actually puts the running head,
        # which is below a naive 8% margin band.
        page.insert_text((72, 93), f"{header} {index + 1}", fontsize=9)
        page.insert_textbox(
            pymupdf.Rect(72, 130, 520, 700), body * 8, fontsize=BODY_FONT_SIZE
        )
    data = doc.tobytes()
    doc.close()
    return data
