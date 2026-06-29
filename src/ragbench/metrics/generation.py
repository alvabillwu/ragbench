"""Generation metrics — measure answer quality without an LLM judge.

These are the "lite" metrics: token overlap with the expected answer (a proxy
for correctness) and faithfulness (does the answer only use tokens present in
the retrieved context — a cheap hallucination proxy). A real LLM-as-judge
backend is pluggable but optional; see ragbench.metrics.judge (future).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..types import Query, RetrievedDoc, Answer, Metric, MetricResult

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


@dataclass
class AnswerOverlap(Metric):
    """Token-set Jaccard similarity between the answer and expected_answer."""

    name: str = "answer_overlap"

    def score(self, query: Query, retrieved: list[RetrievedDoc], answer: Answer) -> MetricResult:
        if not query.expected_answer:
            return MetricResult(self.name, 0.0, {"reason": "no expected answer"})
        a = set(_tokenize(answer.text))
        e = set(_tokenize(query.expected_answer))
        return MetricResult(self.name, _jaccard(a, e), {"answer_tokens": len(a), "expected_tokens": len(e)})


@dataclass
class FaithfulnessLite(Metric):
    """Cheap hallucination proxy: fraction of answer tokens supported by context.

    A token is "supported" if it appears in any retrieved doc. Lower scores
    suggest the answer invents content not in the retrieved context.
    """

    name: str = "faithfulness_lite"

    def score(self, query: Query, retrieved: list[RetrievedDoc], answer: Answer) -> MetricResult:
        ans_tokens = _tokenize(answer.text)
        if not ans_tokens:
            return MetricResult(self.name, 0.0)
        # stopword-ish tokens we don't penalize for being absent from context
        skip = {"a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "and", "or", "it", "this", "that"}
        context_tokens: set[str] = set()
        for d in retrieved:
            context_tokens.update(_tokenize(d.text))

        scored = [t for t in ans_tokens if t not in skip]
        if not scored:
            return MetricResult(self.name, 1.0)
        supported = sum(1 for t in scored if t in context_tokens)
        return MetricResult(
            self.name,
            supported / len(scored),
            {"supported": supported, "scored_tokens": len(scored)},
        )


@dataclass
class CitationCoverage(Metric):
    """Fraction of cited doc ids that were actually retrieved (anti-hallucination)."""

    name: str = "citation_coverage"

    def score(self, query: Query, retrieved: list[RetrievedDoc], answer: Answer) -> MetricResult:
        if not answer.cited_doc_ids:
            return MetricResult(self.name, 0.0, {"reason": "no citations"})
        retrieved_ids = {d.id for d in retrieved}
        valid = sum(1 for cid in answer.cited_doc_ids if cid in retrieved_ids)
        return MetricResult(self.name, valid / len(answer.cited_doc_ids), {"valid": valid})
