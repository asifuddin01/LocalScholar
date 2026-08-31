"""Structured extraction of research details from a single paper.

Summaries, the "Paper Details" panel and the comparison table are all views
over *one* mechanism: pull the fields listed below out of the paper, with a
citation for each. Building three separate features would have meant three
prompts to keep consistent and three ways for them to disagree about what a
paper says.

Two design choices matter:

**Retrieval per field group, not "summarise the whole paper".** A 43-page paper
is 320 chunks. Map-reducing all of them would be hundreds of model calls and
several minutes. Instead each group of fields runs its own scoped retrieval --
"dataset size samples" finds the dataset paragraph directly -- so the model
only ever reads the dozen or so passages that could contain the answer.

**"Not reported" is a first-class value.** Papers genuinely omit things. A
comparison table whose cells are all filled in is a table that has been guessed
at, and guessing is the one thing this tool must not do.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field

from backend.generation.answering import Source, build_sources, format_context
from backend.generation.llm import LLMError, LLMProvider
from backend.retrieval.engine import RetrievalEngine

logger = logging.getLogger(__name__)

NOT_REPORTED = "Not reported"

# A schema is a list of groups. Each group is (name, retrieval query, fields),
# because the query that finds a field is part of the field's definition: you
# cannot extract "loss function" from passages retrieved for "limitations".
#
# Queries deliberately mix vocabularies. "dataset samples participants cohort
# data collection" finds the data section of an ML paper *and* of a genomics
# paper; the original ML-only phrasing returned "Not reported" for the dataset
# of a single-nucleus sequencing study that plainly describes its samples.
FieldGroup = tuple[str, str, tuple[tuple[str, str], ...]]

FIELD_GROUPS: tuple[FieldGroup, ...] = (
    (
        "identity",
        "authors affiliation publication year research problem objective aim of this study",
        (
            ("authors", "Authors"),
            ("year", "Year"),
            ("research_problem", "Research problem"),
            ("task", "Task"),
        ),
    ),
    (
        "data",
        "dataset data samples participants cells cohort size number of samples collected data collection",
        (
            ("dataset", "Dataset"),
            ("dataset_size", "Dataset size"),
        ),
    ),
    (
        "approach",
        "method methods approach model architecture algorithm pipeline analysis training procedure loss function",
        (
            ("model_architecture", "Model / architecture"),
            ("method", "Method"),
            ("training_procedure", "Training procedure"),
            ("loss_function", "Loss function"),
        ),
    ),
    (
        "findings",
        "results findings evaluation metrics performance accuracy we found we observed",
        (
            ("evaluation_metrics", "Evaluation metrics"),
            ("main_results", "Main results"),
        ),
    ),
    (
        "reflection",
        "limitations caveats future work further studies remain to be determined",
        (
            ("limitations", "Limitations"),
            ("future_work", "Future work"),
        ),
    ),
)

# The summary schema produces prose rather than short values. Same machinery,
# same citation verification -- a summary sentence nothing supports is dropped
# exactly like an invented dataset name.
SUMMARY_GROUPS: tuple[FieldGroup, ...] = (
    (
        "problem",
        "research problem motivation objective contribution novelty of this work",
        (
            ("research_problem", "Research problem"),
            ("contribution", "Main contribution"),
        ),
    ),
    (
        "approach",
        "dataset data samples methods approach model experimental setup analysis pipeline",
        (
            ("data", "Data"),
            ("methodology", "Methodology"),
            ("experimental_setup", "Experimental setup"),
        ),
    ),
    (
        "outcome",
        "results findings performance limitations caveats future work",
        (
            ("results", "Results"),
            ("limitations", "Limitations"),
            ("future_work", "Future work"),
        ),
    ),
)

# Title is deliberately absent from FIELD_GROUPS: the parser already read it
# off page one, and asking the model for it produced a confident, plausible,
# entirely invented title ("Privacy-Preserving Ordinal-Meta Learning for Food
# Freshness") for a paper actually called "Leveraging CNN and Random Forest...".
# Never ask a model for something you already know.
ALL_FIELDS: tuple[tuple[str, str], ...] = (("title", "Title"),) + tuple(
    pair for _, _, pairs in FIELD_GROUPS for pair in pairs
)

SUMMARY_SYSTEM_PROMPT = f"""You summarise scientific papers.

You will be given numbered excerpts from ONE paper and a list of summary
sections. Reply with a single JSON object mapping each section name to a
string of one to three complete sentences.

Rules:
- Use ONLY the excerpts. Never use outside knowledge.
- If the excerpts do not cover a section, use exactly "{NOT_REPORTED}".
- Write plain prose. Do not add citation markers; they are attached
  automatically.
- Prefer the paper's own terminology and exact numbers.
- No markdown, no commentary, no code fences."""

SYSTEM_PROMPT = f"""You extract structured facts from scientific papers.

You will be given numbered excerpts from ONE paper and a list of fields.
Reply with a single JSON object mapping each field name to a short string.

Rules:
- Use ONLY the excerpts. Never use outside knowledge.
- If the excerpts do not state a field, use exactly "{NOT_REPORTED}".
  Do not guess, and do not infer a plausible value.
- Keep values short and factual. Copy numbers, dataset names and metric values
  exactly as they appear in the excerpts.
- No markdown, no commentary, no code fences."""

# Deliberately NOT asked for: citations.
#
# The first version of this prompt required {"value": ..., "citations": [...]}
# per field. A 3B model ignores that shape and returns plain strings, so every
# field arrived with no citations and was discarded by the grounding rule --
# even though the evidence was sitting in the excerpts.
#
# Asking the model to cite itself was the wrong design anyway. A citation the
# model reports is a claim about its own reasoning; a citation computed by
# matching the extracted value back against the excerpt text is a check on the
# output. The second one is what a reader actually wants, and it catches
# fabrication rather than trusting the fabricator's word for it.

_STOPWORDS = {
    "the", "and", "for", "with", "was", "were", "are", "this", "that", "from",
    "using", "used", "use", "not", "reported", "which", "their", "they", "has",
    "have", "had", "its", "our", "all", "can", "such", "these", "those", "into",
    "than", "then", "also", "based", "each", "per", "via", "over", "more",
}

# Fraction of a value's distinctive tokens that must appear in one excerpt for
# that excerpt to count as supporting it. Tuned on the four-paper corpus: high
# enough to reject an unrelated passage, low enough to tolerate the model
# rewording ("achieved a score of X" for "score: X").
SUPPORT_THRESHOLD = 0.55
MAX_CITATIONS = 2


def _content_tokens(text: str) -> set[str]:
    """Distinctive tokens, with digit grouping normalised.

    "13,599" and "13599" must match, so commas are stripped from both sides
    before tokenising -- otherwise the single most citable kind of fact in a
    paper, an exact sample count, would never match its own source.
    """
    normalised = text.lower().replace(",", "")
    return {
        token.strip(".")
        for token in re.findall(r"[a-z0-9.]+", normalised)
        if len(token.strip(".")) >= 3 and token.strip(".") not in _STOPWORDS
    }


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def find_supporting_sources_for_prose(
    text: str, sources: list[Source], max_citations: int = 3
) -> list[int]:
    """Verify a multi-sentence passage sentence by sentence.

    A summary paragraph is supported by several different excerpts, so scoring
    the whole paragraph as one bag of tokens dilutes every sentence and finds
    nothing. Each sentence is checked independently and the supporting excerpts
    are unioned.

    A paragraph where no sentence is supported returns nothing, which is what
    makes an unsupported summary indistinguishable from an unsupported fact.
    """
    found: list[int] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        if len(sentence.split()) < 4:
            continue
        for index in find_supporting_sources(sentence, sources):
            if index not in found:
                found.append(index)
    return found[:max_citations]


def find_supporting_sources(value: str, sources: list[Source]) -> list[int]:
    """Which excerpts actually contain this value. Empty means unsupported."""
    tokens = _content_tokens(value)
    if not tokens:
        return []
    # A single-token value ("AdamW", "BCEWithLogitsLoss") is perfectly valid and
    # highly citable; requiring two tokens rejected exactly the short, exact
    # answers this tool exists to find.

    scored: list[tuple[float, int]] = []
    for source in sources:
        overlap = len(tokens & _content_tokens(source.text)) / len(tokens)
        if overlap >= SUPPORT_THRESHOLD:
            scored.append((overlap, source.index))

    scored.sort(reverse=True)
    return [index for _, index in scored[:MAX_CITATIONS]]


@dataclass
class ExtractedField:
    name: str
    label: str
    value: str
    citations: list[int] = field(default_factory=list)

    @property
    def reported(self) -> bool:
        return self.value.strip().lower() not in {NOT_REPORTED.lower(), "", "n/a", "none", "unknown"}


@dataclass
class PaperExtraction:
    document_id: str
    filename: str
    fields: list[ExtractedField]
    sources: list[Source]
    model: str

    def get(self, name: str) -> ExtractedField | None:
        return next((f for f in self.fields if f.name == name), None)

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "model": self.model,
            "fields": [asdict(f) for f in self.fields],
            "sources": [asdict(s) for s in self.sources],
        }


def parse_json_object(raw: str) -> dict:
    """Pull a JSON object out of a model response.

    Small models wrap JSON in code fences or add a sentence of preamble even
    when told not to, so the outermost braces are located rather than trusting
    the response to be clean.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in response")
    return json.loads(text[start:end + 1])


def _coerce(entry: object) -> tuple[str, list[int]]:
    """Accept the shapes models actually produce, not just the one requested."""
    if isinstance(entry, dict):
        value = entry.get("value", entry.get("text", ""))
        citations = entry.get("citations", []) or []
    elif isinstance(entry, (list, tuple)):
        value, citations = ", ".join(str(x) for x in entry), []
    else:
        value, citations = entry, []

    if isinstance(value, (list, tuple)):
        value = ", ".join(str(x) for x in value)
    value = str(value or "").strip() or NOT_REPORTED

    clean_citations: list[int] = []
    if isinstance(citations, (list, tuple)):
        for c in citations:
            try:
                number = int(c)
            except (TypeError, ValueError):
                continue
            if number not in clean_citations:
                clean_citations.append(number)
    return value, clean_citations


class ExtractionService:
    def __init__(
        self,
        engine: RetrievalEngine,
        llm: LLMProvider,
        char_budget: int = 6000,
        # More evidence per group beats sharper evidence per group here: the
        # model is choosing what to report from what it is shown, so breadth of
        # coverage matters more than the ordering within it.
        per_group_k: int = 8,
    ) -> None:
        self.engine = engine
        self.llm = llm
        self.char_budget = char_budget
        self.per_group_k = per_group_k

    def extract(
        self,
        document_id: str,
        filename: str,
        groups: tuple[FieldGroup, ...] = FIELD_GROUPS,
        system_prompt: str = SYSTEM_PROMPT,
        prose: bool = False,
    ) -> PaperExtraction:
        """Run a schema against one paper.

        `prose=True` switches to sentence-level citation verification, which is
        what summary sections need; short factual values are verified whole.
        """
        all_sources: list[Source] = []
        results: list[ExtractedField] = []
        source_offset = 0

        for _, query, fields in groups:
            chunks = self.engine.hybrid_search(
                query,
                top_k=self.per_group_k,
                document_ids=[document_id],
                # Measured both ways on a four-paper corpus. Reranking here made
                # extraction 2-3x slower and, net, slightly *worse*: 34 of 60
                # fields filled against 41 without it. It helped one paper and
                # hurt two, which is within the run-to-run variance of a 3B
                # model. Question answering still reranks, because there
                # precision at rank 1 decides the answer outright.
                use_reranker=False,
            )
            sources = build_sources(chunks, self.char_budget)
            # Each group is numbered independently for the model, then shifted
            # into a document-wide numbering so citations stay unique overall.
            for source in sources:
                source.index += source_offset

            group_fields = self._extract_group(
                query, fields, sources, system_prompt=system_prompt, prose=prose
            )
            results.extend(group_fields)
            all_sources.extend(sources)
            source_offset += len(sources)

        return PaperExtraction(
            document_id=document_id, filename=filename, fields=results,
            sources=all_sources, model=self.llm.model,
        )

    def _extract_group(
        self,
        query: str,
        fields: tuple[tuple[str, str], ...],
        sources: list[Source],
        system_prompt: str = SYSTEM_PROMPT,
        prose: bool = False,
    ) -> list[ExtractedField]:
        blank = [ExtractedField(name=n, label=l, value=NOT_REPORTED) for n, l in fields]
        if not sources:
            return blank

        field_list = "\n".join(f'- "{name}": {label}' for name, label in fields)
        instruction = (
            "Write these summary sections as JSON:" if prose
            else "Extract these fields as JSON:"
        )
        prompt = (
            f"Excerpts:\n\n{format_context(sources)}\n\n"
            f"{instruction}\n{field_list}\n\n"
            f"JSON object only."
        )

        try:
            payload = parse_json_object(
                self.llm.complete(system_prompt, prompt, json_mode=True)
            )
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            # One bad group must not lose the other two. The fields come back
            # as "Not reported", which is both true and visibly distinguishable
            # from a value the model made up.
            logger.warning("Extraction failed for group %r: %s", query[:40], exc)
            return blank

        extracted: list[ExtractedField] = []
        for name, label in fields:
            if name not in payload:
                extracted.append(ExtractedField(name=name, label=label, value=NOT_REPORTED))
                continue

            value, _ = _coerce(payload[name])
            if value.strip().lower() == NOT_REPORTED.lower():
                extracted.append(ExtractedField(name=name, label=label, value=NOT_REPORTED))
                continue

            # Citations are computed, not taken on trust: an excerpt only counts
            # if it actually contains the value the model produced.
            citations = (
                find_supporting_sources_for_prose(value, sources) if prose
                else find_supporting_sources(value, sources)
            )

            # A value no excerpt supports did not come from the paper. That is
            # how the invented title and authors were caught, and it is the
            # whole reason this tool can be trusted for literature review.
            if not citations:
                logger.info("Dropping unsupported value for %r: %r", name, value[:60])
                value = NOT_REPORTED

            extracted.append(
                ExtractedField(name=name, label=label, value=value, citations=citations)
            )
        return extracted
