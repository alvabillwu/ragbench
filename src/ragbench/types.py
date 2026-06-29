"""Core data types — the contracts a RAG pipeline plugs into.

A "pipeline" in ragbench is just two callables:
  - a retriever: (query_text) -> list[RetrievedDoc]
  - a generator: (query_text, list[RetrievedDoc]) -> Answer
These types pin down what those callables exchange, so any RAG stack
(LangChain, LlamaIndex, hand-rolled) can be wrapped to fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol


@dataclass(frozen=True)
class Query:
    """A benchmark question with its ground truth."""

    id: str
    text: str
    # Ground-truth answer text (for answer-similarity metrics). May be empty.
    expected_answer: str = ""
    # Set of doc ids that a perfect retriever would return, in priority order.
    relevant_doc_ids: tuple[str, ...] = ()
    # Free-form metadata (e.g. difficulty, category) for slicing results.
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedDoc:
    """One document returned by the retriever."""

    id: str
    text: str
    # Relevance score the retriever assigned (higher = more relevant). Optional.
    score: float = 0.0


@dataclass(frozen=True)
class Answer:
    """The generator's response."""

    text: str
    # Which retrieved doc ids the answer claims to draw from (for citation checks).
    cited_doc_ids: tuple[str, ...] = ()
    # Latency in seconds, if the pipeline measures it.
    latency_seconds: float = 0.0


@dataclass(frozen=True)
class RunResult:
    """One query's full execution record — input, output, and metric scores."""

    query: Query
    retrieved: tuple[RetrievedDoc, ...]
    answer: Answer
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Convenience: did this query clear a default bar on its primary metric."""
        return self.scores.get("overall", 0.0) >= 0.5


@dataclass(frozen=True)
class Dataset:
    """A named, versioned collection of queries."""

    name: str
    version: str
    queries: tuple[Query, ...]

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self):
        return iter(self.queries)


# ── Pipeline protocols ──────────────────────────────────────────────────────


class Retriever(Protocol):
    def __call__(self, query_text: str) -> list[RetrievedDoc]: ...


class Generator(Protocol):
    def __call__(self, query_text: str, docs: list[RetrievedDoc]) -> Answer: ...


# Plain callable aliases for users who prefer them.
RetrieverFn = Callable[[str], list[RetrievedDoc]]
GeneratorFn = Callable[[str, list[RetrievedDoc]], Answer]


# ── Metric protocol ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    detail: dict = field(default_factory=dict)


class Metric(Protocol):
    """A metric scores a single (query, retrieved, answer) triple."""

    name: str

    def score(
        self,
        query: Query,
        retrieved: list[RetrievedDoc],
        answer: Answer,
    ) -> MetricResult: ...
