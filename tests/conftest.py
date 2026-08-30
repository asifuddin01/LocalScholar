"""Shared fixtures.

Tests build their PDFs in memory rather than committing binary fixtures, so
the exact layout under test (page breaks, headings, font sizes) is visible in
the test file itself.
"""

from __future__ import annotations

import pymupdf
import pytest

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


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient with its data directory pointed at a temp folder."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("LOCALSCHOLAR_DATA_DIR", str(tmp_path / "data"))
    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client
