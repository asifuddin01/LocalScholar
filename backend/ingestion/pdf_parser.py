"""PDF -> page-aware, section-aware document representation.

This module is deliberately the most careful part of the ingestion path.
Everything downstream (chunking, retrieval, and above all citations) can only
be as good as the two fields produced here:

    page_number  -- so a citation can say "page 4"
    section      -- so a citation can say "Section: Dataset"

If we lost those at parse time, no amount of retrieval quality could recover
them. So the parser works at *line* granularity, tracks headings as it goes,
and handles the layout quirk that matters most for research papers: two-column
text, which a naive top-to-bottom read order interleaves into nonsense.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)


# --- Tunables ---------------------------------------------------------------
# These are heuristics, not truths. They are module constants rather than YAML
# config because changing them requires understanding the code below; exposing
# them as user settings would imply a tuning story we don't have.

HEADING_MAX_CHARS = 90          # headings are short; body paragraphs are not
HEADING_SIZE_RATIO = 1.25       # a heading is >=25% larger than body text
TITLE_REGION_RATIO = 0.45       # top fraction of page 1 holding title + authors
NUMBERED_HEADING_MAX_CHARS = 60 # "III. METHODOLOGY" is short; list items are not
NUMBERED_HEADING_MAX_WORDS = 8  # ditto -- a heading is a label, not a sentence
COLUMN_TOLERANCE_RATIO = 0.02   # slack when deciding which column a block is in
MIN_CHARS_PER_PAGE = 40         # below this average, we assume a scanned PDF
ROW_OVERLAP_RATIO = 0.5         # vertical overlap that makes two lines "a row"

COLUMN_FULL, COLUMN_LEFT, COLUMN_RIGHT = 0, 1, 2

# "3", "3.", "3.1", "A.", "IV." followed by a word -> almost certainly a heading.
NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:\d{1,2}(?:\.\d{1,2}){0,3}|[A-Z]|[IVXL]{1,5})[.)]?\s+(?=[A-Za-z])"
)

# Canonical research-paper sections. Matching one of these is strong evidence,
# independent of font size -- which matters because plenty of papers set their
# headings at body size and rely on bold alone.
KNOWN_SECTIONS = {
    "abstract", "introduction", "background", "related work", "prior work",
    "motivation", "preliminaries", "problem statement", "problem formulation",
    "method", "methods", "methodology", "approach", "proposed method",
    "materials and methods", "model", "architecture", "network architecture",
    "data", "dataset", "datasets", "data and preprocessing", "preprocessing",
    "implementation", "implementation details", "training", "training details",
    "experimental setup", "experiments", "experimental results", "setup",
    "evaluation", "evaluation metrics", "metrics", "results",
    "results and discussion", "ablation", "ablation study", "ablation studies",
    "analysis", "discussion", "comparison", "baselines",
    "limitations", "conclusion", "conclusions", "conclusion and future work",
    "future work", "acknowledgments", "acknowledgements", "references",
    "bibliography", "appendix", "supplementary material",
}

# Leading numbering to strip before comparing against KNOWN_SECTIONS.
_LEADING_NUMBER_RE = re.compile(r"^\s*(?:\d{1,2}(?:\.\d{1,2}){0,3}|[A-Z]|[IVXL]{1,5})[.)]?\s+")
_BARE_PAGE_NUMBER_RE = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")

# IEEE and similar templates run these into the paragraph they label
# ("Abstract—In recent years, ...") rather than giving them a heading line.
# The abstract is high-value text for summarisation, so it is worth labelling.
_INLINE_SECTION_RE = re.compile(
    r"^\s*(abstract|index terms|keywords)\s*[—–\-:.]", re.IGNORECASE
)
_INLINE_SECTION_NAMES = {
    "abstract": "Abstract",
    "index terms": "Index Terms",
    "keywords": "Keywords",
}
_WHITESPACE_RE = re.compile(r"\s+")

# A line holding nothing but a section number. Springer/LNCS templates put the
# number in its own text line, separate from the heading words beside it.
_BARE_ENUMERATOR_RE = re.compile(
    r"^\s*(?:\d{1,2}(?:\.\d{1,2}){0,3}|[A-Z]|[IVXL]{1,5})[.)]?\s*$"
)


class PDFParseError(Exception):
    """Raised when a PDF cannot be turned into usable text."""


@dataclass
class TextBlock:
    """A paragraph-sized run of text with the metadata a citation needs."""

    text: str
    page_number: int          # 1-indexed, matches what a PDF reader displays
    section: str | None
    is_heading: bool = False
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass
class ParsedPage:
    page_number: int
    text: str
    blocks: list[TextBlock] = field(default_factory=list)


@dataclass
class ParsedDocument:
    title: str | None
    page_count: int
    pages: list[ParsedPage]
    sections: list[str]

    @property
    def blocks(self) -> list[TextBlock]:
        return [b for page in self.pages for b in page.blocks]

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)


# --- Internal line representation -------------------------------------------


@dataclass
class _Line:
    text: str
    size: float       # largest font size on the line
    bold: bool
    bbox: tuple[float, float, float, float]


def _line_from_dict(line: dict) -> _Line | None:
    """Build a line record, or None if there is nothing readable on it.

    Two subtleties, both of which silently corrupt text if missed:

    1. PyMuPDF emits inter-word spaces as their own spans. Filtering spans by
       ``.strip()`` before joining glues every word together
       ("LeveragingCNNandRandomForest"), which destroys tokenisation for BM25
       and readability for the LLM. So we join *all* spans and only use the
       inked ones to measure font size and weight.
    2. Rotated text (the arXiv stamp down the left margin, watermarks) is not
       part of the prose and pollutes both headings and chunks, so we skip any
       line that is not horizontal.
    """
    direction = line.get("dir", (1.0, 0.0))
    if abs(float(direction[0]) - 1.0) > 0.01 or abs(float(direction[1])) > 0.01:
        return None

    spans = line.get("spans", [])
    if not spans:
        return None
    text = "".join(s.get("text", "") for s in spans)
    inked = [s for s in spans if s.get("text", "").strip()]
    if not text.strip() or not inked:
        return None

    size = max(float(s.get("size", 0.0)) for s in inked)
    inked_chars = sum(len(s["text"]) for s in inked)
    # PyMuPDF packs style into a bitfield; bit 4 (value 16) is bold.
    bold_chars = sum(len(s["text"]) for s in inked if int(s.get("flags", 0)) & 16)
    bold = bold_chars > inked_chars * 0.6
    return _Line(text=text, size=size, bold=bold, bbox=tuple(line["bbox"]))


def _body_font_size(doc: pymupdf.Document, sample_pages: int = 8) -> float:
    """The most common font size, weighted by how much text is set in it.

    Weighting by character count is what makes this robust: a paper with many
    large headings still has far more *characters* in body text.
    """
    counter: Counter[float] = Counter()
    for page in doc[: min(sample_pages, doc.page_count)]:
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        counter[round(float(span.get("size", 0.0)), 1)] += len(text)
    if not counter:
        return 10.0
    return counter.most_common(1)[0][0]


def _order_blocks(blocks: list[dict], page_width: float) -> list[tuple[int, dict]]:
    """Return blocks in human reading order, handling two-column layouts.

    A naive sort by vertical position reads a two-column paper as
    "left line 1, right line 1, left line 2, ..." which destroys every
    sentence. We detect the two-column case and read each column in full.

    Returns (column_id, block) pairs. The column id is not just bookkeeping:
    heading detection uses it to tell a heading apart from a table cell, which
    only makes sense *within* a column.
    """
    if not blocks:
        return []

    mid = page_width / 2
    tol = page_width * COLUMN_TOLERANCE_RATIO
    left: list[dict] = []
    right: list[dict] = []
    full: list[dict] = []

    for b in blocks:
        x0, _, x1, _ = b["bbox"]
        if x1 <= mid + tol:
            left.append(b)
        elif x0 >= mid - tol:
            right.append(b)
        else:
            full.append(b)

    by_y = lambda b: (b["bbox"][1], b["bbox"][0])  # noqa: E731

    # Need real content on both sides before we believe it's two columns.
    if len(left) >= 2 and len(right) >= 2:
        column_top = min(b["bbox"][1] for b in left + right)
        header = [b for b in full if b["bbox"][3] <= column_top]
        # Full-width blocks lower down are usually spanning figures or tables.
        # Reading them after both columns keeps prose contiguous; the page
        # number is unaffected, so citations stay correct either way.
        rest = [b for b in full if b["bbox"][3] > column_top]
        return (
            [(COLUMN_FULL, b) for b in sorted(header, key=by_y)]
            + [(COLUMN_LEFT, b) for b in sorted(left, key=by_y)]
            + [(COLUMN_RIGHT, b) for b in sorted(right, key=by_y)]
            + [(COLUMN_FULL, b) for b in sorted(rest, key=by_y)]
        )

    return [(COLUMN_FULL, b) for b in sorted(blocks, key=by_y)]


def _merge_enumerator_lines(
    lines: list[tuple[int, int, "_Line"]]
) -> list[tuple[int, int, "_Line"]]:
    """Rejoin section numbers that the PDF stores as separate lines.

    Springer/LNCS papers lay out "1  Introduction" as two text lines: a lone
    "1", and "Introduction" beside it. Left alone this breaks heading detection
    twice over -- the number looks like a table cell sitting next to the title,
    which suppresses the heading, and the recovered label loses its number.
    Merging them first makes both problems disappear.
    """
    merged_out: list[tuple[int, int, _Line]] = []
    consumed: set[int] = set()

    for i, (column, block_index, line) in enumerate(lines):
        if i in consumed:
            continue
        if not _BARE_ENUMERATOR_RE.match(_clean(line.text)):
            merged_out.append((column, block_index, line))
            continue

        partner: int | None = None
        for j, (other_column, _, other) in enumerate(lines):
            if j == i or j in consumed or other_column != column:
                continue
            if other.bbox[0] < line.bbox[2]:          # must sit to the right
                continue
            overlap = min(line.bbox[3], other.bbox[3]) - max(line.bbox[1], other.bbox[1])
            shortest = min(line.bbox[3] - line.bbox[1], other.bbox[3] - other.bbox[1])
            if shortest <= 0 or overlap < shortest * ROW_OVERLAP_RATIO:
                continue
            if other.bbox[0] - line.bbox[2] > line.size * 2:   # too far to be one heading
                continue
            if partner is None or other.bbox[0] < lines[partner][2].bbox[0]:
                partner = j

        if partner is None:
            merged_out.append((column, block_index, line))
            continue

        _, partner_block, other = lines[partner]
        consumed.add(partner)
        merged_out.append((
            column,
            partner_block,
            _Line(
                text=f"{_clean(line.text)} {other.text}",
                size=max(line.size, other.size),
                bold=line.bold or other.bold,
                bbox=(
                    min(line.bbox[0], other.bbox[0]), min(line.bbox[1], other.bbox[1]),
                    max(line.bbox[2], other.bbox[2]), max(line.bbox[3], other.bbox[3]),
                ),
            ),
        ))
    return merged_out


def _mark_row_neighbours(lines: list[tuple[int, int, "_Line"]]) -> list[bool]:
    """Flag lines that sit beside another line in the same column.

    This is how we tell the heading "Model" apart from a table column header
    "Model". Typography can't: plenty of templates (IEEE's among them) set
    every heading at body size with no bold weight, so font metrics give no
    signal at all. Layout does. A heading owns its line; a table cell has
    siblings to its left or right at the same height.

    Comparisons are restricted to the same column, otherwise every heading in
    a two-column paper would look like a table cell because of the text beside
    it in the other column.
    """
    flags = [False] * len(lines)
    by_column: dict[int, list[int]] = {}
    for index, (column, _, _) in enumerate(lines):
        by_column.setdefault(column, []).append(index)

    for indices in by_column.values():
        indices.sort(key=lambda i: lines[i][2].bbox[1])
        for position, i in enumerate(indices):
            a = lines[i][2].bbox
            # Sorted by top edge, so we only need to look forward until a line
            # starts below where this one ends.
            for k in range(position + 1, len(indices)):
                j = indices[k]
                b = lines[j][2].bbox
                if b[1] >= a[3]:
                    break
                if not (b[0] > a[2] + 2 or b[2] < a[0] - 2):
                    continue                      # not horizontally disjoint
                if _BARE_ENUMERATOR_RE.match(_clean(lines[j][2].text)):
                    continue                      # a stray section number, not a cell
                overlap = min(a[3], b[3]) - max(a[1], b[1])
                # Measured against each line's *own* height, not the shorter of
                # the two. Papers are full of superscripts and inline math that
                # PyMuPDF reports as very short lines; judged by the shorter
                # height, one stray superscript beside a heading would overlap
                # it "fully" and suppress a real section.
                height_a = a[3] - a[1]
                height_b = b[3] - b[1]
                if height_a > 0 and overlap > height_a * ROW_OVERLAP_RATIO:
                    flags[i] = True
                if height_b > 0 and overlap > height_b * ROW_OVERLAP_RATIO:
                    flags[j] = True
    return flags


def _clean(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _is_numbered_heading(text: str) -> bool:
    """Is this a numbered section heading, as opposed to a numbered list item?

    The bare "number followed by words" pattern is far too permissive on real
    papers. It fires on enumerated contributions ("1. We developed six novel
    models..."), on reference entries ("2. Tamim N, Elshrkawey M..."), and even
    on body text that happens to wrap onto a line starting with a decimal
    ("0.78. While its precision..."). Each of those would then become the
    running section label for the text that follows it.

    A genuine heading is a short label: a clean section number, a capitalised
    title of a few words, and no trailing hyphen from a wrapped line.
    """
    match = NUMBERED_HEADING_RE.match(text)
    if not match:
        return False
    if len(text) > NUMBERED_HEADING_MAX_CHARS:
        return False
    if text.endswith("-"):          # a hyphen means the line was wrapped mid-word
        return False

    marker = match.group(0).strip()
    if marker[:1] == "0":           # section numbers don't start at zero; decimals do
        return False

    rest = text[match.end():].strip()
    if not rest or not rest[:1].isupper():
        return False
    if len(rest.split()) > NUMBERED_HEADING_MAX_WORDS:
        return False
    # "1) Selection: The FL server chooses a subset..." is a labelled paragraph.
    if ":" in rest and len(rest.split(":", 1)[1].split()) > 2:
        return False
    return True


def _looks_like_heading(
    line: _Line,
    body_size: float,
    *,
    in_title_region: bool,
    in_references: bool,
    has_row_neighbour: bool,
) -> bool:
    """Decide whether a line starts a new section.

    Deliberately biased toward precision. A missed heading is cosmetic -- the
    text keeps the previous section label. A *false* heading is worse: it
    becomes the running section for everything after it, so one bold table cell
    can mislabel the rest of a page and every citation drawn from it. Bold-only
    detection was tried and dropped for exactly this reason: table headers in
    research papers are bold, short, and capitalised.

    So a heading must be one of three high-confidence signals:
      (a) a canonical research-paper section name,
      (b) a numbered heading ("3.1 Dataset", "IV. RESULTS"), or
      (c) set markedly larger than body text.

    Inside the reference list all of this is switched off except (a).
    Bibliographies have no subsections, but they are full of short capitalised
    lines with leading numbers and initials, which trip every other rule.
    """
    text = _clean(line.text)
    if not text or len(text) > HEADING_MAX_CHARS:
        return False
    if _BARE_PAGE_NUMBER_RE.match(text):
        return False
    if not any(ch.isalpha() for ch in text):
        return False
    if has_row_neighbour:       # something sits beside it: a table row, not a heading
        return False

    # Body text ending in sentence punctuation is a sentence, not a heading.
    # Checked first: without it, a wrapped body line ending in "...the data."
    # normalises to "data" and matches a canonical section name.
    if text.endswith((".", ",", ";", "?", "!")):
        return False

    # (a) Canonical section name, with any numbering stripped.
    normalized = _LEADING_NUMBER_RE.sub("", text).strip().lower().rstrip(":")
    if normalized in KNOWN_SECTIONS and text[:1].isupper():
        return True

    if in_references:
        return False

    # (b) Numbered heading. Reliable in papers, but not in the title block,
    # where affiliation lines are sometimes numbered footnote markers.
    if not in_title_region and _is_numbered_heading(text):
        return True

    # (c) A real jump in font size. Suppressed inside the title region, where
    # the paper title and author names are the largest text on the page and
    # would otherwise be mistaken for section headings.
    if not in_title_region and line.size >= body_size * HEADING_SIZE_RATIO:
        return True

    return False


def _join_lines(lines: list[str]) -> str:
    """Join wrapped lines back into a paragraph, undoing hyphenation."""
    out = ""
    for raw in lines:
        piece = raw.strip()
        if not piece:
            continue
        if not out:
            out = piece
            continue
        if out.endswith("-") and piece[:1].islower():
            out = out[:-1] + piece      # "seg-" + "mentation" -> "segmentation"
        else:
            out = f"{out} {piece}"
    return _clean(out)


def _extract_title(doc: pymupdf.Document, body_size: float) -> str | None:
    """Prefer embedded metadata, fall back to the largest text on page one."""
    meta_title = _clean((doc.metadata or {}).get("title") or "")
    if len(meta_title) > 8 and not meta_title.lower().endswith((".pdf", ".dvi", ".tex")):
        return meta_title

    if doc.page_count == 0:
        return None

    page = doc[0]
    height = page.rect.height
    candidates: list[_Line] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for raw_line in block.get("lines", []):
            line = _line_from_dict(raw_line)
            # Titles live in the top 45% of page one and are set large.
            if line and line.bbox[1] < height * 0.45 and line.size > body_size * 1.15:
                candidates.append(line)
    if not candidates:
        return None

    largest = max(c.size for c in candidates)
    title_lines = [c for c in candidates if c.size >= largest - 0.5]
    title_lines.sort(key=lambda c: c.bbox[1])
    title = _join_lines([c.text for c in title_lines])
    return title or None


def parse_pdf(path: str | Path) -> ParsedDocument:
    """Parse a PDF into pages of section-tagged text blocks.

    Raises PDFParseError for encrypted, corrupt, or image-only (scanned) PDFs
    rather than silently returning empty text -- a document that indexes to
    nothing would otherwise look successful and then answer every question
    with "no evidence found".
    """
    path = Path(path)
    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # noqa: BLE001 - PyMuPDF raises a variety of types
        raise PDFParseError(f"Could not open {path.name}: {exc}") from exc

    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise PDFParseError(f"{path.name} is password-protected.")
        if doc.page_count == 0:
            raise PDFParseError(f"{path.name} contains no pages.")

        body_size = _body_font_size(doc)
        title = _extract_title(doc, body_size)

        pages: list[ParsedPage] = []
        sections: list[str] = []
        current_section: str | None = None
        in_references = False

        for page_index in range(doc.page_count):
            page = doc[page_index]
            page_number = page_index + 1
            raw = page.get_text("dict")
            text_blocks = [b for b in raw.get("blocks", []) if b.get("type") == 0]
            ordered = _order_blocks(text_blocks, page.rect.width)

            # First pass: flatten to lines, remembering which column and which
            # PDF block each came from. Block index is the paragraph boundary.
            line_entries: list[tuple[int, int, _Line]] = []
            for block_index, (column, block) in enumerate(ordered):
                for raw_line in block.get("lines", []):
                    line = _line_from_dict(raw_line)
                    if line is not None:
                        line_entries.append((column, block_index, line))
            line_entries = _merge_enumerator_lines(line_entries)
            row_neighbour = _mark_row_neighbours(line_entries)

            page_blocks: list[TextBlock] = []
            buffer: list[str] = []
            buffer_bbox: list[tuple[float, float, float, float]] = []

            def flush() -> None:
                nonlocal buffer, buffer_bbox
                if not buffer:
                    return
                text = _join_lines(buffer)
                # Single stray characters and page furniture add noise to BM25
                # and waste embedding calls.
                if len(text) >= 3 and not _BARE_PAGE_NUMBER_RE.match(text):
                    x0 = min(b[0] for b in buffer_bbox)
                    y0 = min(b[1] for b in buffer_bbox)
                    x1 = max(b[2] for b in buffer_bbox)
                    y1 = max(b[3] for b in buffer_bbox)
                    page_blocks.append(
                        TextBlock(
                            text=text,
                            page_number=page_number,
                            section=current_section,
                            bbox=(x0, y0, x1, y1),
                        )
                    )
                buffer = []
                buffer_bbox = []

            title_region_cutoff = page.rect.height * TITLE_REGION_RATIO
            previous_block_index = -1

            # Second pass: walk the lines in reading order, tracking sections.
            for entry_index, (_, block_index, line) in enumerate(line_entries):
                if block_index != previous_block_index:
                    flush()  # a PDF block boundary is a paragraph boundary
                    previous_block_index = block_index

                text = _clean(line.text)
                in_title_region = (
                    page_number == 1 and line.bbox[1] < title_region_cutoff
                )

                inline = _INLINE_SECTION_RE.match(text)
                if inline:
                    # Label the section but keep the line's text: the words
                    # after "Abstract—" are the abstract itself.
                    flush()
                    current_section = _INLINE_SECTION_NAMES[inline.group(1).lower()]
                    if current_section not in sections:
                        sections.append(current_section)
                    buffer.append(line.text)
                    buffer_bbox.append(tuple(line.bbox))
                    continue

                if _looks_like_heading(
                    line,
                    body_size,
                    in_title_region=in_title_region,
                    in_references=in_references,
                    has_row_neighbour=row_neighbour[entry_index],
                ):
                    flush()
                    current_section = text.rstrip(":")
                    in_references = (
                        _LEADING_NUMBER_RE.sub("", current_section)
                        .strip().lower().rstrip(":.")
                        in {"references", "bibliography"}
                    )
                    if current_section not in sections:
                        sections.append(current_section)
                    page_blocks.append(
                        TextBlock(
                            text=current_section,
                            page_number=page_number,
                            section=current_section,
                            is_heading=True,
                            bbox=tuple(line.bbox),
                        )
                    )
                else:
                    buffer.append(line.text)
                    buffer_bbox.append(tuple(line.bbox))
            flush()

            page_text = "\n\n".join(b.text for b in page_blocks if not b.is_heading)
            pages.append(
                ParsedPage(page_number=page_number, text=page_text, blocks=page_blocks)
            )

        parsed = ParsedDocument(
            title=title, page_count=doc.page_count, pages=pages, sections=sections
        )

        if parsed.char_count < MIN_CHARS_PER_PAGE * doc.page_count:
            raise PDFParseError(
                f"{path.name} yielded almost no selectable text "
                f"({parsed.char_count} characters across {doc.page_count} pages). "
                "It is probably a scanned document; LocalScholar does not run OCR."
            )

        logger.info(
            "Parsed %s: %d pages, %d blocks, %d sections",
            path.name, parsed.page_count, len(parsed.blocks), len(sections),
        )
        return parsed
    finally:
        doc.close()
