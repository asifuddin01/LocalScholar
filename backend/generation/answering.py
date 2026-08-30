"""Grounded question answering.

The contract this module enforces is narrow and deliberate:

  * the model sees only text that was actually retrieved from the user's PDFs;
  * every source it sees is numbered, and it must cite those numbers;
  * a citation that does not correspond to a retrieved chunk is stripped after
    the fact, so a fabricated "[7]" cannot reach the user;
  * if the evidence does not support an answer, the answer is "I could not find
    sufficient evidence", not a plausible guess from the model's own memory.

The last point is the one that makes this usable for literature review. A tool
that quietly answers from pretraining when the paper is silent is worse than no
tool, because you cannot tell the two cases apart.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from backend.generation.llm import LLMError, LLMProvider
from backend.retrieval.vector_store import ScoredChunk

logger = logging.getLogger(__name__)

NOT_FOUND_TOKEN = "INSUFFICIENT_EVIDENCE"
NOT_FOUND_MESSAGE = (
    "I could not find sufficient evidence in the uploaded papers to answer this question."
)

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")

SYSTEM_PROMPT = f"""You are a careful research assistant. You answer questions about \
scientific papers using ONLY the numbered sources provided to you.

Rules, in order of importance:
1. Use ONLY information present in the sources. Never use outside knowledge, and never \
infer a value that is not written down.
2. Cite the source number in square brackets after each claim, like [1] or [2][3]. Every \
factual sentence must carry at least one citation.
3. If the sources do not contain enough information to answer, reply with exactly \
{NOT_FOUND_TOKEN} and nothing else. Do not apologise or speculate.
4. If different papers disagree, say so and cite both.
5. Quote exact numbers, dataset names, and metric values as they appear in the sources.
6. Be concise. Two to five sentences unless the question asks for a list."""


@dataclass
class Source:
    """One numbered piece of evidence put in front of the model."""

    index: int
    chunk_id: str
    document_id: str
    filename: str
    title: str | None
    page_number: int
    section: str | None
    text: str
    score: float


@dataclass
class Answer:
    answer: str
    sources: list[Source]
    cited_indexes: list[int]
    found: bool
    model: str


def build_sources(chunks: list[ScoredChunk], char_budget: int) -> list[Source]:
    """Number the retrieved chunks, stopping at the context budget."""
    sources: list[Source] = []
    used = 0
    for chunk in chunks:
        if used + len(chunk.text) > char_budget and sources:
            break
        sources.append(
            Source(
                index=len(sources) + 1,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                title=chunk.title,
                page_number=chunk.page_number,
                section=chunk.section,
                text=chunk.text,
                score=chunk.score,
            )
        )
        used += len(chunk.text)
    return sources


def format_context(sources: list[Source]) -> str:
    """Render sources for the prompt.

    The provenance line is included so the model can attribute a claim to a
    specific paper when several are in play -- multi-paper questions are
    unanswerable if every excerpt looks anonymous.
    """
    blocks = []
    for source in sources:
        label = source.filename
        if source.section:
            label += f" — {source.section}"
        blocks.append(
            f"[{source.index}] {label} (page {source.page_number})\n{source.text}"
        )
    return "\n\n".join(blocks)


def extract_citations(text: str, valid: set[int]) -> tuple[str, list[int]]:
    """Keep citations that point at a real source; delete the rest.

    Small models occasionally emit a citation number beyond the range they were
    given. Rendering it would show the user a source that does not exist, which
    is precisely the failure this project promises not to have.
    """
    dropped: list[str] = []

    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if number in valid:
            return match.group(0)
        dropped.append(match.group(0))
        return ""

    cleaned = _CITATION_RE.sub(replace, text)
    if dropped:
        logger.warning("Removed %d invented citation(s): %s", len(dropped), dropped)

    cited: list[int] = []
    for match in _CITATION_RE.finditer(cleaned):
        number = int(match.group(1))
        if number not in cited:
            cited.append(number)

    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    # Small models sometimes open with the citation ("[1] The optimizer was...").
    # It reads as a footnote marker for a sentence that has not happened yet.
    cleaned = re.sub(r"^\s*(?:\[\d{1,2}\]\s*)+", "", cleaned)
    return cleaned.strip(), cited


class AnswerService:
    def __init__(self, llm: LLMProvider, char_budget: int = 6000) -> None:
        self.llm = llm
        self.char_budget = char_budget

    def answer(self, question: str, chunks: list[ScoredChunk]) -> Answer:
        # No evidence retrieved at all: don't even ask the model. There is
        # nothing it could answer from except its own memory.
        if not chunks:
            return Answer(
                answer=NOT_FOUND_MESSAGE, sources=[], cited_indexes=[],
                found=False, model=self.llm.model,
            )

        sources = build_sources(chunks, self.char_budget)
        prompt = (
            f"Sources:\n\n{format_context(sources)}\n\n"
            f"Question: {question}\n\n"
            f"Answer using only the sources above, citing them as [n]."
        )

        raw = self.llm.complete(SYSTEM_PROMPT, prompt)

        if NOT_FOUND_TOKEN in raw.upper():
            return Answer(
                answer=NOT_FOUND_MESSAGE, sources=sources, cited_indexes=[],
                found=False, model=self.llm.model,
            )

        cleaned, cited = extract_citations(raw, {s.index for s in sources})

        # An answer with no surviving citation is ungrounded by definition:
        # either the model ignored the format, or every citation it produced
        # was invented. Either way it must not be presented as evidence-backed.
        if not cited:
            logger.warning("Model produced an answer with no valid citations; rejecting.")
            return Answer(
                answer=NOT_FOUND_MESSAGE, sources=sources, cited_indexes=[],
                found=False, model=self.llm.model,
            )

        return Answer(
            answer=cleaned, sources=sources, cited_indexes=cited,
            found=True, model=self.llm.model,
        )
