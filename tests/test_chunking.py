"""Chunking tests.

The invariants here are what make citations checkable. If a chunk can span two
pages, then a citation built from it can only say "somewhere around page 4".
"""

from __future__ import annotations

from backend.ingestion.chunking import ChunkingConfig, chunk_document
from backend.ingestion.pdf_parser import ParsedDocument, ParsedPage, TextBlock


def make_parsed(blocks: list[TextBlock], title: str | None = "T") -> ParsedDocument:
    pages: dict[int, ParsedPage] = {}
    for block in blocks:
        page = pages.setdefault(
            block.page_number, ParsedPage(page_number=block.page_number, text="")
        )
        page.blocks.append(block)
    ordered = [pages[n] for n in sorted(pages)]
    sections = list(dict.fromkeys(b.section for b in blocks if b.section))
    return ParsedDocument(
        title=title, page_count=len(ordered), pages=ordered, sections=sections
    )


def block(text, page, section=None, heading=False):
    return TextBlock(text=text, page_number=page, section=section, is_heading=heading)


def test_chunk_never_spans_two_pages():
    """The invariant that makes a citation point at one page."""
    parsed = make_parsed([
        block("alpha " * 30, 1, "Method"),
        block("bravo " * 30, 2, "Method"),   # same section, next page
    ])
    chunks = chunk_document("doc", "p.pdf", parsed, ChunkingConfig(chunk_size=4000))
    assert len(chunks) == 2
    assert {c.page_number for c in chunks} == {1, 2}
    for chunk in chunks:
        assert not ("alpha" in chunk.text and "bravo" in chunk.text)


def test_chunk_never_spans_two_sections():
    parsed = make_parsed([
        block("alpha " * 20, 1, "Dataset"),
        block("bravo " * 20, 1, "Method"),   # same page, next section
    ])
    chunks = chunk_document("doc", "p.pdf", parsed, ChunkingConfig(chunk_size=4000))
    assert [c.section for c in chunks] == ["Dataset", "Method"]


def test_long_section_is_split_near_the_configured_size():
    parsed = make_parsed([block("word " * 600, 1, "Method")])
    config = ChunkingConfig(chunk_size=400, overlap=50)
    chunks = chunk_document("doc", "p.pdf", parsed, config)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= config.chunk_size + config.min_chunk_size


def test_headings_are_not_indexed_as_content():
    parsed = make_parsed([
        block("Dataset", 1, "Dataset", heading=True),
        block("We use KiTS19, which has 300 scans. " * 4, 1, "Dataset"),
    ])
    chunks = chunk_document("doc", "p.pdf", parsed)
    assert len(chunks) == 1
    assert not chunks[0].text.startswith("Dataset\n")


def test_reference_sections_are_excluded():
    parsed = make_parsed([
        block("Real finding about the dataset. " * 4, 1, "Dataset"),
        block("[1] Smith et al. A paper about things. " * 4, 9, "References"),
        block("Acknowledged funding sources here. " * 4, 9, "Acknowledgements"),
    ])
    chunks = chunk_document("doc", "p.pdf", parsed)
    assert [c.section for c in chunks] == ["Dataset"]


def test_numbered_reference_heading_is_also_excluded():
    parsed = make_parsed([block("[1] Smith et al. " * 8, 9, "8. References")])
    assert chunk_document("doc", "p.pdf", parsed) == []


def test_search_text_carries_the_section_but_displayed_text_does_not():
    """Evidence shown to the user must be verbatim; the prefix is index-only."""
    parsed = make_parsed([block("We use the KiTS19 dataset. " * 5, 4, "3.1 Dataset")])
    chunk = chunk_document("doc", "p.pdf", parsed)[0]
    assert chunk.search_text.startswith("3.1 Dataset")
    assert not chunk.text.startswith("3.1 Dataset")
    assert chunk.text in chunk.search_text


def test_overlap_does_not_start_mid_word():
    """A chunk starting on "ctices" is noise in the index and looks broken."""
    parsed = make_parsed([block("practices " * 200, 1, "Method")])
    chunks = chunk_document("doc", "p.pdf", parsed, ChunkingConfig(chunk_size=300, overlap=60))
    assert len(chunks) > 1
    for chunk in chunks[1:]:
        assert chunk.text.split()[0] in {"practices"}


def test_short_trailing_fragment_is_folded_into_the_previous_chunk():
    parsed = make_parsed([
        block("word " * 100, 1, "Method"),
        block("A short tail.", 1, "Method"),
    ])
    chunks = chunk_document("doc", "p.pdf", parsed, ChunkingConfig(chunk_size=450, min_chunk_size=200))
    assert "A short tail." in chunks[-1].text
    assert len(chunks[-1].text) > 200


def test_chunk_ids_are_unique_and_ordinals_sequential():
    parsed = make_parsed([block("word " * 100, page, "Method") for page in (1, 2, 3)])
    chunks = chunk_document("doc42", "p.pdf", parsed)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.chunk_id.startswith("doc42:") for c in chunks)


def test_metadata_is_attached_to_every_chunk():
    parsed = make_parsed([block("word " * 60, 7, "4 Results")], title="My Paper")
    chunk = chunk_document("doc", "paper.pdf", parsed)[0]
    assert (chunk.document_id, chunk.filename, chunk.page_number) == ("doc", "paper.pdf", 7)
    assert (chunk.section, chunk.title) == ("4 Results", "My Paper")
