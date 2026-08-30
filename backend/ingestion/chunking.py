"""Chunking: parsed pages -> retrievable units.

Three rules shape this, and each costs something elsewhere:

1. **A chunk never spans a page.** This is the expensive one -- it splits the
   occasional paragraph at a page break -- and it is worth it, because it means
   every chunk has exactly one page number. A chunk straddling pages 4 and 5
   can only ever produce a vague citation, and a vague citation is one the
   reader cannot check.

2. **A chunk never spans a section.** Sections are the unit researchers think
   in. Mixing the end of "Dataset" with the start of "Method" produces a chunk
   that is a good match for neither question.

3. **Chunks are built from whole paragraphs where possible.** The parser
   already recovered paragraph boundaries, so there is no reason to cut
   mid-sentence at a fixed character count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.ingestion.pdf_parser import ParsedDocument, TextBlock

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """One retrievable unit, carrying everything a citation needs."""

    chunk_id: str
    document_id: str
    filename: str
    text: str                 # verbatim, shown to the user as evidence
    page_number: int
    section: str | None
    ordinal: int              # position within the document
    title: str | None = None

    @property
    def search_text(self) -> str:
        """The text that gets embedded and indexed, as opposed to displayed.

        The section heading is prepended because a chunk pulled out of its
        document loses the context that made it meaningful: a paragraph reading
        "300 scans were used, split 80/20" answers "what dataset size?" only if
        the retriever can see it sits under "Dataset". The verbatim `text` is
        what gets shown as evidence, so this prefix never leaks into a citation.
        """
        if self.section:
            return f"{self.section}\n\n{self.text}"
        return self.text


@dataclass
class ChunkingConfig:
    chunk_size: int = 800
    overlap: int = 120
    min_chunk_size: int = 200
    # Reference lists match query terms constantly (every paper name, every
    # author, every venue) while containing no findings of their own. Indexing
    # them buries real evidence under citations of other people's work.
    exclude_sections: tuple[str, ...] = (
        "references", "bibliography", "acknowledgments", "acknowledgements",
    )


_LEADING_NUMBER_RE = re.compile(
    r"^\s*(?:\d{1,2}(?:\.\d{1,2}){0,3}|[A-Z]|[IVXL]{1,5})[.)]?\s+"
)


def _is_excluded(section: str | None, config: ChunkingConfig) -> bool:
    if not section:
        return False
    normalized = _LEADING_NUMBER_RE.sub("", section).strip().lower().rstrip(":.")
    return normalized in config.exclude_sections


def _tail_overlap(text: str, overlap: int) -> str:
    """Take the last ~`overlap` characters, snapped to a sentence boundary.

    Overlap exists so a fact split across a chunk boundary is still findable in
    at least one whole chunk. Cutting it mid-sentence would defeat that, so the
    tail starts at the last sentence break that fits.
    """
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    sentences = _SENTENCE_END_RE.split(tail, maxsplit=1)
    if len(sentences) > 1:
        return sentences[-1].strip()
    # No sentence break in range. Fall back to a word boundary -- slicing by
    # raw character count leaves the next chunk starting on "ctices", which is
    # noise in the index and looks like a bug when shown as evidence.
    space = tail.find(" ")
    return tail[space + 1:].strip() if space != -1 else tail.strip()


def _hard_split(text: str, size: int) -> list[str]:
    """Last-resort split on word boundaries.

    Needed because sentence splitting alone is not a guarantee. Author lists,
    table rows flattened into a line, equation blocks and reference strings all
    routinely run for thousands of characters without a single full stop. Left
    unsplit, one of those becomes a chunk that swamps the whole LLM context
    budget on its own.
    """
    pieces: list[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + len(word) + 1 > size:
            pieces.append(current)
            current = ""
        current = f"{current} {word}".strip() if current else word
    if current:
        pieces.append(current)
    return pieces


def _split_long_paragraph(text: str, config: ChunkingConfig) -> list[str]:
    """Break a paragraph that exceeds the budget on its own, at sentence ends."""
    if len(text) <= config.chunk_size:
        return [text]

    units: list[str] = []
    for sentence in _SENTENCE_END_RE.split(text):
        if not sentence:
            continue
        if len(sentence) > config.chunk_size:
            units.extend(_hard_split(sentence, config.chunk_size))
        else:
            units.append(sentence)

    pieces: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) + 1 > config.chunk_size:
            pieces.append(current.strip())
            current = _tail_overlap(current, config.overlap)
        current = f"{current} {unit}".strip()
    if current.strip():
        pieces.append(current.strip())
    return pieces


def chunk_document(
    document_id: str,
    filename: str,
    parsed: ParsedDocument,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Turn a parsed document into chunks, preserving page and section."""
    config = config or ChunkingConfig()
    chunks: list[Chunk] = []
    ordinal = 0

    def emit(text: str, page_number: int, section: str | None) -> None:
        nonlocal ordinal
        text = text.strip()
        if len(text) < 40:      # page furniture, stray captions, orphan lines
            return
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}:{ordinal}",
                document_id=document_id,
                filename=filename,
                text=text,
                page_number=page_number,
                section=section,
                ordinal=ordinal,
                title=parsed.title,
            )
        )
        ordinal += 1

    # Group blocks into runs sharing one page and one section. Rules 1 and 2
    # fall out of this grouping rather than needing checks further down.
    groups: list[tuple[int, str | None, list[TextBlock]]] = []
    for block in parsed.blocks:
        if block.is_heading or _is_excluded(block.section, config):
            continue
        if groups and groups[-1][0] == block.page_number and groups[-1][1] == block.section:
            groups[-1][2].append(block)
        else:
            groups.append((block.page_number, block.section, [block]))

    for page_number, section, blocks in groups:
        buffer = ""
        for block in blocks:
            for piece in _split_long_paragraph(block.text, config):
                if buffer and len(buffer) + len(piece) + 2 > config.chunk_size:
                    emit(buffer, page_number, section)
                    buffer = _tail_overlap(buffer, config.overlap)
                buffer = f"{buffer}\n\n{piece}".strip() if buffer else piece

        if not buffer:
            continue
        # A short remainder is folded back into the previous chunk of the same
        # page and section rather than emitted as a stub that is too small to
        # answer anything on its own.
        if (
            len(buffer) < config.min_chunk_size
            and chunks
            and chunks[-1].page_number == page_number
            and chunks[-1].section == section
        ):
            merged = f"{chunks[-1].text}\n\n{buffer}".strip()
            if len(merged) <= config.chunk_size + config.min_chunk_size:
                chunks[-1].text = merged
                continue
        emit(buffer, page_number, section)

    return chunks
