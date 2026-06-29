"""Report aggregation — turn a BenchmarkRun into a scorecard + summary."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .runner import BenchmarkRun
from .types import RunResult


@dataclass
class Scorecard:
    """Aggregate scores across all queries in a run."""

    dataset_name: str
    n_queries: int
    metric_means: dict[str, float] = field(default_factory=dict)
    metric_stdev: dict[str, float] = field(default_factory=dict)
    overall_mean: float = 0.0
    pass_rate: float = 0.0  # fraction of queries with overall >= 0.5
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "n_queries": self.n_queries,
            "overall_mean": round(self.overall_mean, 4),
            "pass_rate": round(self.pass_rate, 4),
            "metric_means": {k: round(v, 4) for k, v in self.metric_means.items()},
            "metric_stdev": {k: round(v, 4) for k, v in self.metric_stdev.items()},
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def scorecard(run: BenchmarkRun) -> Scorecard:
    """Aggregate a run's per-query scores into means + stdevs."""
    if not run.results:
        return Scorecard(dataset_name=run.dataset_name, n_queries=0)

    # Collect all metric names seen (union, since some metrics skip queries).
    metric_names: set[str] = set()
    for r in run.results:
        metric_names.update(r.scores.keys())

    means: dict[str, float] = {}
    stdevs: dict[str, float] = {}
    for name in metric_names:
        values = [r.scores.get(name, 0.0) for r in run.results]
        means[name] = statistics.fmean(values)
        stdevs[name] = statistics.pstdev(values) if len(values) > 1 else 0.0

    overalls = [r.scores.get("overall", 0.0) for r in run.results]
    pass_rate = sum(1 for o in overalls if o >= 0.5) / len(overalls)

    return Scorecard(
        dataset_name=run.dataset_name,
        n_queries=len(run.results),
        metric_means=means,
        metric_stdev=stdevs,
        overall_mean=statistics.fmean(overalls),
        pass_rate=pass_rate,
        elapsed_seconds=run.elapsed_seconds,
    )


def render_scorecard(sc: Scorecard) -> str:
    """A compact human-readable scorecard."""
    lines = [
        f"ragbench — {sc.dataset_name} ({sc.n_queries} queries, {sc.elapsed_seconds:.2f}s)",
        f"  overall: {sc.overall_mean:.3f}   pass-rate: {sc.pass_rate:.1%}",
        "  metrics:",
    ]
    for name in sorted(sc.metric_means.keys()):
        m = sc.metric_means[name]
        s = sc.metric_stdev.get(name, 0.0)
        bar = "█" * int(m * 20)
        lines.append(f"    {name:<20} {m:.3f} ±{s:.3f}  {bar}")
    return "\n".join(lines)
