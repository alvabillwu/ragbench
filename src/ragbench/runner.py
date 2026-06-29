"""The benchmark runner — executes a pipeline over a dataset and collects scores."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .types import (
    Dataset,
    Query,
    RetrievedDoc,
    Answer,
    RunResult,
    Metric,
    RetrieverFn,
    GeneratorFn,
)
from .metrics import PrecisionAtK, RecallAtK, MRR, NDCGAtK
from .metrics.generation import AnswerOverlap, FaithfulnessLite, CitationCoverage


def default_metrics() -> list[Metric]:
    """The standard metric suite: 4 retrieval + 3 generation metrics."""
    return [
        PrecisionAtK(k=5),
        RecallAtK(k=5),
        MRR(),
        NDCGAtK(k=5),
        AnswerOverlap(),
        FaithfulnessLite(),
        CitationCoverage(),
    ]


@dataclass
class BenchmarkConfig:
    """How the runner executes and scores a pipeline."""

    metrics: list[Metric] = field(default_factory=default_metrics)
    # Weight of each metric when computing the overall score (0..1, normalized).
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "precision@k": 0.15,
            "recall@k": 0.15,
            "mrr": 0.15,
            "ndcg@k": 0.15,
            "answer_overlap": 0.2,
            "faithfulness_lite": 0.15,
            "citation_coverage": 0.05,
        }
    )
    # If True, surface the first exception from the pipeline instead of scoring 0.
    fail_fast: bool = False


def _overall(scores: dict[str, float], weights: dict[str, float]) -> float:
    total_w = 0.0
    total = 0.0
    for name, w in weights.items():
        if name in scores:
            total += scores[name] * w
            total_w += w
    return total / total_w if total_w else 0.0


def run_single(
    query: Query,
    retriever: RetrieverFn,
    generator: GeneratorFn,
    config: BenchmarkConfig,
) -> RunResult:
    """Run one query through retriever + generator and score it."""
    try:
        retrieved = retriever(query.text)
    except Exception:
        if config.fail_fast:
            raise
        retrieved = []
    try:
        answer = generator(query.text, retrieved)
    except Exception:
        if config.fail_fast:
            raise
        answer = Answer(text="")

    scores: dict[str, float] = {}
    for metric in config.metrics:
        result = metric.score(query, retrieved, answer)
        scores[result.name] = result.value
    scores["overall"] = _overall(scores, config.weights)
    return RunResult(query=query, retrieved=tuple(retrieved), answer=answer, scores=scores)


@dataclass
class BenchmarkRun:
    """The full result of running a pipeline over a dataset."""

    dataset_name: str
    dataset_version: str
    config: BenchmarkConfig
    results: list[RunResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def __len__(self) -> int:
        return len(self.results)


def run_benchmark(
    dataset: Dataset,
    retriever: RetrieverFn,
    generator: GeneratorFn,
    config: Optional[BenchmarkConfig] = None,
) -> BenchmarkRun:
    """Run a retriever+generator pipeline over every query in a dataset."""
    config = config or BenchmarkConfig()
    start = time.perf_counter()
    results: list[RunResult] = []
    for q in dataset:
        results.append(run_single(q, retriever, generator, config))
    elapsed = time.perf_counter() - start
    return BenchmarkRun(
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        config=config,
        results=results,
        elapsed_seconds=elapsed,
    )
