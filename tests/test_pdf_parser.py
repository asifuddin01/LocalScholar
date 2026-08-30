"""Parser tests.

The assertions here are mostly about *metadata survival*. Extracting text from
a PDF is the easy half; keeping the page number and section attached to it is
what makes a citation verifiable, so that is what these tests guard.
"""

from __future__ import annotations

import pytest

from backend.ingestion.pdf_parser import (
    PDFParseError,
    _clean,
    _is_numbered_heading,
    _join_lines,
    _order_blocks,
    parse_pdf,
)
from tests.conftest import build_blank_pdf, build_paper_pdf


def test_parses_page_count_and_title(paper_pdf):
    doc = parse_pdf(paper_pdf)
    assert doc.page_count == 2
    assert doc.title == "A Study of Local Retrieval"


def test_every_block_carries_a_valid_page_number(paper_pdf):
    doc = parse_pdf(paper_pdf)
    assert doc.blocks, "expected some text blocks"
    for block in doc.blocks:
        assert 1 <= block.page_number <= doc.page_count


def test_page_numbers_are_one_indexed_and_correct(paper_pdf):
    """A citation that says "page 2" must mean what a PDF reader calls page 2."""
    doc = parse_pdf(paper_pdf)
    text_by_page = {page.page_number: page.text for page in doc.pages}
    assert "Prior work" in text_by_page[1]
    assert "KiTS19" in text_by_page[2]
    assert "KiTS19" not in text_by_page[1]


def test_detects_sections(paper_pdf):
    doc = parse_pdf(paper_pdf)
    assert doc.sections == ["Abstract", "1 Introduction", "2 Methods", "3 Results"]


def test_body_text_inherits_the_enclosing_section(paper_pdf):
    doc = parse_pdf(paper_pdf)
    dataset_blocks = [b for b in doc.blocks if "KiTS19" in b.text and not b.is_heading]
    assert dataset_blocks, "expected the methods paragraph"
    assert dataset_blocks[0].section == "2 Methods"
    assert dataset_blocks[0].page_number == 2


def test_scanned_pdf_is_rejected_rather_than_silently_empty(tmp_path):
    path = tmp_path / "scanned.pdf"
    path.write_bytes(build_blank_pdf())
    with pytest.raises(PDFParseError, match="scanned"):
        parse_pdf(path)


def test_missing_file_raises_parse_error(tmp_path):
    with pytest.raises(PDFParseError):
        parse_pdf(tmp_path / "nope.pdf")


def test_join_lines_undoes_hyphenation():
    assert _join_lines(["seg-", "mentation of the"]) == "segmentation of the"
    assert _join_lines(["a well-", "Known model"]) == "a well- Known model"
    assert _join_lines(["first line", "second line"]) == "first line second line"


@pytest.mark.parametrize("text", [
    "III. METHODOLOGY",
    "3.1 Dataset",
    "A. Dataset Description",
])
def test_accepts_real_numbered_headings(text):
    assert _is_numbered_heading(text)


@pytest.mark.parametrize("text", [
    "0.78. While its precision and recall were competitive for",   # wrapped body text
    "2.0 and JSIEC datasets in our analysis as it is the latest",  # decimal, lowercase
    "1. We developed six novel models that combine deeper CNN backbones, trans-",
    "1) Selection: The FL server chooses a subset of connected",   # labelled paragraph
])
def test_rejects_numbered_text_that_is_not_a_heading(text):
    assert not _is_numbered_heading(text)


def test_two_column_pages_are_read_column_by_column():
    """The layout bug that matters most: interleaving two columns line by line."""
    page_width = 600.0
    blocks = [
        {"bbox": (50, 100, 280, 130), "id": "left-top"},
        {"bbox": (320, 100, 550, 130), "id": "right-top"},
        {"bbox": (50, 140, 280, 170), "id": "left-bottom"},
        {"bbox": (320, 140, 550, 170), "id": "right-bottom"},
    ]
    order = [block["id"] for _, block in _order_blocks(blocks, page_width)]
    assert order == ["left-top", "left-bottom", "right-top", "right-bottom"]


def test_single_column_pages_are_read_top_to_bottom():
    page_width = 600.0
    blocks = [
        {"bbox": (50, 200, 550, 240), "id": "second"},
        {"bbox": (50, 100, 550, 140), "id": "first"},
    ]
    order = [block["id"] for _, block in _order_blocks(blocks, page_width)]
    assert order == ["first", "second"]


def test_clean_collapses_whitespace():
    assert _clean("  a   b \n c ") == "a b c"
