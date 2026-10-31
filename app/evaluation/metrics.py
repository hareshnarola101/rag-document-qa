"""Simple, transparent RAG metrics.

- hit_rate@k: did any of the top-k chunks contain the evidence for the question?
- MRR: 1 / rank of the first chunk containing the evidence (0 if none).
- faithfulness: share of answer sentences whose content words are mostly present in the
  cited chunks. It's lexical, so it catches answers that drift away from their sources but
  can't judge paraphrases; an LLM judge would be the next step.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

from app.core.text import content_words, split_sentences

_CITATION_MARK_RE = re.compile(r"\[\d+\]")


@dataclass(frozen=True, slots=True)
class EvalItem:
    question: str
    source: str  # filename of the document that answers the question
    evidence: str  # a phrase that a correct chunk must contain


@dataclass(frozen=True, slots=True)
class QuestionResult:
    question: str
    relevance: list[bool]  # per retrieved chunk, in rank order
    faithfulness: float | None  # None when the system declined to answer
    answered: bool


def _squash(text: str) -> str:
    return " ".join(text.split()).lower()


def contains_evidence(item: EvalItem, filename: str, content: str) -> bool:
    return filename == item.source and _squash(item.evidence) in _squash(content)


def hit_at_k(relevance: Sequence[bool], k: int) -> float:
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: Sequence[bool]) -> float:
    for rank, relevant in enumerate(relevance, start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def faithfulness(answer: str, contexts: Sequence[str], min_support: float = 0.8) -> float | None:
    """Fraction of answer sentences supported by the contexts; None if there's nothing to score."""
    sentences = split_sentences(_CITATION_MARK_RE.sub("", answer))
    scored = [set(content_words(s)) for s in sentences]
    scored = [terms for terms in scored if terms]
    if not scored:
        return None

    context_terms = {term for context in contexts for term in content_words(context)}
    supported = sum(1 for terms in scored if len(terms & context_terms) / len(terms) >= min_support)
    return supported / len(scored)


@dataclass(frozen=True, slots=True)
class EvalSummary:
    questions: int
    hit_rate: dict[int, float]
    mrr: float
    faithfulness: float | None
    answer_rate: float


def summarize(results: Sequence[QuestionResult], ks: Sequence[int] = (1, 3, 5)) -> EvalSummary:
    if not results:
        raise ValueError("no results to summarize")
    scores = [r.faithfulness for r in results if r.faithfulness is not None]
    return EvalSummary(
        questions=len(results),
        hit_rate={k: fmean(hit_at_k(r.relevance, k) for r in results) for k in ks},
        mrr=fmean(reciprocal_rank(r.relevance) for r in results),
        faithfulness=fmean(scores) if scores else None,
        answer_rate=fmean(1.0 if r.answered else 0.0 for r in results),
    )
