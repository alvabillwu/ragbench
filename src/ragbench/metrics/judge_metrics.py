"""Judge-based metrics — use a JudgeBackend to score faithfulness / relevance.

These complement the deterministic lite metrics: where FaithfulnessLite is a
cheap token-overlap proxy, FaithfulnessJudge asks a real (or mock) judge
"is every claim in the answer supported by the context?". Same Metric protocol,
so they drop straight into a BenchmarkConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import Query, RetrievedDoc, Answer, Metric, MetricResult
from ..judge import JudgeBackend, MockJudge


@dataclass
class FaithfulnessJudge(Metric):
    """Judge-scored faithfulness: is the answer grounded in the context?"""

    judge: JudgeBackend = field(default_factory=MockJudge)
    name: str = "faithfulness_judge"

    def score(self, query: Query, retrieved: list[RetrievedDoc], answer: Answer) -> MetricResult:
        if not answer.text:
            return MetricResult(self.name, 0.0, {"reason": "empty answer"})
        v = self.judge.judge(
            "Score whether the answer is faithful to and fully grounded in the context (1.0 = fully grounded, 0.0 = unsupported claims).",
            query,
            retrieved,
            answer,
        )
        return MetricResult(self.name, v.score, {"rationale": v.rationale})


@dataclass
class AnswerRelevanceJudge(Metric):
    """Judge-scored answer relevance: does the answer address the question?"""

    judge: JudgeBackend = field(default_factory=MockJudge)
    name: str = "answer_relevance_judge"

    def score(self, query: Query, retrieved: list[RetrievedDoc], answer: Answer) -> MetricResult:
        if not answer.text:
            return MetricResult(self.name, 0.0, {"reason": "empty answer"})
        v = self.judge.judge(
            "Score how relevant the answer is to the question (1.0 = directly answers it, 0.0 = irrelevant).",
            query,
            retrieved,
            answer,
        )
        return MetricResult(self.name, v.score, {"rationale": v.rationale})
