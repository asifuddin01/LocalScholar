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

# (field name, human label) grouped by the retrieval that finds them.
FIELD_GROUPS: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    (
        "identity",
        "title authors publication year research problem objective and task addressed",
        (
            ("authors", "Authors"),
            ("year", "Year"),
            ("research_problem", "Research problem"),
            ("task", "Task"),
        ),
    ),
    (
        "approach",
        "dataset name number of samples size model architecture method training procedure loss function",
        (
            ("dataset", "Dataset"),
            ("dataset_size", "Dataset size"),
            ("model_architecture", "Model / architecture"),
            ("method", "Method"),
            ("training_procedure", "Training procedure"),
            ("loss_function", "Loss function"),
        ),
    ),
    (
        "findings",
        "evaluation metrics main results baselines limitations future work",
        (
            ("evaluation_metrics", "Evaluation metrics"),
            ("main_results", "Main results"),
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

SYSTEM_PROMPT = f"""You extract structured facts from scientific papers.

You will be given numbered excerpts from ONE paper and a list of fields.
Reply with a single JSON object and nothing else.

Rules:
- Use ONLY the excerpts. Never use outside knowledge.
- Every value must be an object: {{"value": "...", "citations": [1, 2]}}
- Citations are the numbers of the excerpts that support the value.
- If the excerpts do not state a field, use {{"value": "{NOT_REPORTED}", "citations": []}}.
  Do not guess, and do not infer a plausible value.
- Keep values short and factual. Quote numbers exactly as written.
- No markdown, no commentary, no code fences."""


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
        per_group_k: int = 6,
    ) -> None:
        self.engine = engine
        self.llm = llm
        self.char_budget = char_budget
        self.per_group_k = per_group_k

    def extract(self, document_id: str, filename: str) -> PaperExtraction:
        all_sources: list[Source] = []
        results: list[ExtractedField] = []
        source_offset = 0

        for _, query, fields in FIELD_GROUPS:
            chunks = self.engine.hybrid_search(
                query,
                top_k=self.per_group_k,
                document_ids=[document_id],
                # ~2.5s per group on CPU, for candidates that are already scoped
                # to one paper and one topic. Not worth it here; question
                # answering, where precision at rank 1 decides the answer,
                # still uses it.
                use_reranker=False,
            )
            sources = build_sources(chunks, self.char_budget)
            # Each group is numbered independently for the model, then shifted
            # into a document-wide numbering so citations stay unique overall.
            for source in sources:
                source.index += source_offset

            group_fields = self._extract_group(query, fields, sources)
            results.extend(group_fields)
            all_sources.extend(sources)
            source_offset += len(sources)

        return PaperExtraction(
            document_id=document_id, filename=filename, fields=results,
            sources=all_sources, model=self.llm.model,
        )

    def _extract_group(
        self, query: str, fields: tuple[tuple[str, str], ...], sources: list[Source]
    ) -> list[ExtractedField]:
        blank = [ExtractedField(name=n, label=l, value=NOT_REPORTED) for n, l in fields]
        if not sources:
            return blank

        field_list = "\n".join(f'- "{name}": {label}' for name, label in fields)
        prompt = (
            f"Excerpts:\n\n{format_context(sources)}\n\n"
            f"Extract these fields as JSON:\n{field_list}\n\n"
            f"JSON object only."
        )

        try:
            payload = parse_json_object(self.llm.complete(SYSTEM_PROMPT, prompt))
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            # One bad group must not lose the other two. The fields come back
            # as "Not reported", which is both true and visibly distinguishable
            # from a value the model made up.
            logger.warning("Extraction failed for group %r: %s", query[:40], exc)
            return blank

        valid_indexes = {s.index for s in sources}
        extracted: list[ExtractedField] = []
        for name, label in fields:
            if name not in payload:
                extracted.append(ExtractedField(name=name, label=label, value=NOT_REPORTED))
                continue
            value, citations = _coerce(payload[name])
            # Same rule as question answering: a citation that does not point at
            # a retrieved excerpt is dropped, not shown.
            citations = [c for c in citations if c in valid_indexes]

            # And the rule that matters most: a value with no surviving citation
            # is not evidence-backed, so it is not reported. This is how the
            # hallucinated author/year values get caught -- the model produced
            # them confidently but could not point at a single excerpt, because
            # no excerpt said it.
            if not citations:
                value = NOT_REPORTED

            extracted.append(
                ExtractedField(name=name, label=label, value=value, citations=citations)
            )
        return extracted
