"""ragbench — lightweight RAG pipeline benchmarking.

Design goals (this is a medium-complexity project, built incrementally):
  1. Synthetic datasets so you can benchmark without a private corpus.
  2. Deterministic retrieval + generation metrics (no LLM calls needed for core).
  3. Pluggable LLM-as-judge for faithfulness/relevance (optional, needs an API).
  4. Reproducible: every run is a versioned, hashable artifact.

Module map:
  ragbench.types     — core dataclasses (Query, RetrievedDoc, Answer, RunResult)
  ragbench.datasets  — synthetic dataset generators
  ragbench.metrics   — metric implementations (retrieval + generation)
  ragbench.runner    — runs a pipeline over a dataset, collects results
  ragbench.report    — aggregates results into a scorecard
"""

from .types import Query, RetrievedDoc, Answer, RunResult, Dataset, Metric, MetricResult
from .runner import run_benchmark, BenchmarkConfig, default_metrics, metrics_with_judge
from .report import scorecard
from .judge import JudgeBackend, MockJudge, LLMJudge, Verdict, get_judge
from .diff import diff_runs, render_diff, DiffReport

__version__ = "0.3.0"
__all__ = [
    "Query",
    "RetrievedDoc",
    "Answer",
    "RunResult",
    "Dataset",
    "Metric",
    "MetricResult",
    "run_benchmark",
    "BenchmarkConfig",
    "default_metrics",
    "metrics_with_judge",
    "scorecard",
    "JudgeBackend",
    "MockJudge",
    "LLMJudge",
    "Verdict",
    "get_judge",
    "diff_runs",
    "render_diff",
    "DiffReport",
]
