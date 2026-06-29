"""Retrieval metrics — measure how well the retriever surfaces relevant docs.

All metrics are deterministic and need no LLM calls. They assume each Query
carries `relevant_doc_ids` (the ground-truth relevant set).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..types import Query, RetrievedDoc, Answer, Metric, MetricResult


@dataclass
class PrecisionAtK(Metric):
    """Fraction of the top-K retrieved docs that are relevant."""

    k: int = 5
    name: str = "precision@k"

    def score(self, query: Query, retrieved: list[RetrievedDoc], answer: Answer) -> MetricResult:
        if not query.relevant_doc_ids:
            return MetricResult(self.name, 0.0, {"reason": "no ground truth"})
        top = retrieved[: self.k]
        if not top:
            return MetricResult(self.name, 0.0)
        rel = set(query.relevant_doc_ids)
        hits = sum(1 for d in top if d.id in rel)
        return MetricResult(self.name, hits / len(top), {"k": self.k, "hits": hits})


@dataclass
class RecallAtK(Metric):
    """Fraction of relevant docs that appear in the top-K retrieved."""

    k: int = 5
    name: str = "recall@k"

    def score(self, query: Query, retrieved: list[RetrievedDoc], answer: Answer) -> MetricResult:
        relevant = set(query.relevant_doc_ids)
        if not relevant:
            return MetricResult(self.name, 0.0, {"reason": "no ground truth"})
        top_ids = {d.id for d in retrieved[: self.k]}
        hits = len(relevant & top_ids)
        return MetricResult(self.name, hits / len(relevant), {"k": self.k, "hits": hits})


@dataclass
class MRR(Metric):
    """Mean Reciprocal Rank — 1/rank of the first relevant doc."""

    name: str = "mrr"

    def score(self, query: Query, retrieved: list[RetrievedDoc], answer: Answer) -> MetricResult:
        relevant = set(query.relevant_doc_ids)
        if not relevant or not retrieved:
            return MetricResult(self.name, 0.0)
        for rank, d in enumerate(retrieved, start=1):
            if d.id in relevant:
                return MetricResult(self.name, 1.0 / rank, {"first_hit_rank": rank})
        return MetricResult(self.name, 0.0)


@dataclass
class NDCGAtK(Metric):
    """Normalized Discounted Cumulative Gain at K.

    Treats the ground-truth relevant set as binary relevance (rel=1 if in set).
    """

    k: int = 5
    name: str = "ndcg@k"

    def score(self, query: Query, retrieved: list[RetrievedDoc], answer: Answer) -> MetricResult:
        relevant = set(query.relevant_doc_ids)
        if not relevant or not retrieved:
            return MetricResult(self.name, 0.0)

        top = retrieved[: self.k]

        def dcg(rels: list[float]) -> float:
            return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))

        gains = [1.0 if d.id in relevant else 0.0 for d in top]
        actual = dcg(gains)
        # Ideal: all relevant docs ranked first, capped at k.
        ideal_n = min(len(relevant), self.k)
        ideal = dcg([1.0] * ideal_n)
        if ideal == 0:
            return MetricResult(self.name, 0.0)
        return MetricResult(self.name, actual / ideal, {"k": self.k})
