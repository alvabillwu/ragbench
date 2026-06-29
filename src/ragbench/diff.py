"""Run-diffing — compare two BenchmarkRuns head-to-head.

Given two runs over the same dataset (e.g. pipeline A vs pipeline B, or the
same pipeline with two configs), produce a per-metric delta report: which
pipeline wins each metric, by how much, and which individual queries flipped.
This is the "A/B" view of benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .runner import BenchmarkRun
from .report import scorecard


@dataclass
class MetricDelta:
    name: str
    a_mean: float
    b_mean: float
    delta: float  # a - b (positive => A better)
    winner: str  # "A" | "B" | "tie"

    def as_dict(self) -> dict:
        return {
            "metric": self.name,
            "a": round(self.a_mean, 4),
            "b": round(self.b_mean, 4),
            "delta": round(self.delta, 4),
            "winner": self.winner,
        }


@dataclass
class QueryFlip:
    """A query where the per-query overall winner differs between A and B."""

    query_id: str
    a_overall: float
    b_overall: float
    winner: str  # "A" | "B" | "tie"


@dataclass
class DiffReport:
    """Head-to-head comparison of two runs."""

    a_label: str
    b_label: str
    dataset: str
    n_queries: int
    metric_deltas: list[MetricDelta] = field(default_factory=list)
    query_flips: list[QueryFlip] = field(default_factory=list)
    overall_delta: float = 0.0  # A overall_mean - B overall_mean
    overall_winner: str = "tie"
    a_wins: int = 0  # metrics where A wins
    b_wins: int = 0

    def as_dict(self) -> dict:
        return {
            "a": self.a_label,
            "b": self.b_label,
            "dataset": self.dataset,
            "n_queries": self.n_queries,
            "overall_delta": round(self.overall_delta, 4),
            "overall_winner": self.overall_winner,
            "metric_wins": {"A": self.a_wins, "B": self.b_wins},
            "metrics": [m.as_dict() for m in self.metric_deltas],
            "query_flips": len(self.query_flips),
        }


def _winner(a: float, b: float, eps: float = 1e-9) -> str:
    if a - b > eps:
        return "A"
    if b - a > eps:
        return "B"
    return "tie"


def diff_runs(
    run_a: BenchmarkRun,
    run_b: BenchmarkRun,
    a_label: str = "A",
    b_label: str = "B",
) -> DiffReport:
    """Compare two BenchmarkRuns over the (assumed same) dataset.

    Pairs results by query position. If the runs differ in length, only the
    common prefix is compared and the count reflects that.
    """
    sc_a = scorecard(run_a)
    sc_b = scorecard(run_b)

    # Union of metric names across both runs.
    metric_names = sorted(set(sc_a.metric_means) | set(sc_b.metric_means))
    deltas: list[MetricDelta] = []
    a_wins = b_wins = 0
    for name in metric_names:
        am = sc_a.metric_means.get(name, 0.0)
        bm = sc_b.metric_means.get(name, 0.0)
        delta = am - bm
        w = _winner(am, bm)
        if w == "A":
            a_wins += 1
        elif w == "B":
            b_wins += 1
        deltas.append(MetricDelta(name=name, a_mean=am, b_mean=bm, delta=delta, winner=w))

    # Per-query flips on overall score.
    n = min(len(run_a.results), len(run_b.results))
    flips: list[QueryFlip] = []
    for i in range(n):
        ra = run_a.results[i]
        rb = run_b.results[i]
        oa = ra.scores.get("overall", 0.0)
        ob = rb.scores.get("overall", 0.0)
        w = _winner(oa, ob)
        # A "flip" is a query where there's a clear winner (not a tie) — i.e. the
        # pipelines disagree on that query.
        if w != "tie":
            flips.append(QueryFlip(query_id=ra.query.id, a_overall=oa, b_overall=ob, winner=w))

    overall_delta = sc_a.overall_mean - sc_b.overall_mean
    report = DiffReport(
        a_label=a_label,
        b_label=b_label,
        dataset=run_a.dataset_name,
        n_queries=n,
        metric_deltas=deltas,
        query_flips=flips,
        overall_delta=overall_delta,
        overall_winner=_winner(sc_a.overall_mean, sc_b.overall_mean),
        a_wins=a_wins,
        b_wins=b_wins,
    )
    return report


def render_diff(report: DiffReport) -> str:
    """Human-readable A/B comparison."""
    lines = [
        f"ragbench A/B — {report.a_label} vs {report.b_label}  ({report.dataset}, {report.n_queries} queries)",
        f"  overall winner: {report.overall_winner}  (delta {report.overall_delta:+.4f})",
        f"  metric wins: {report.a_label}={report.a_wins}  {report.b_label}={report.b_wins}",
        "",
        f"  {'metric':<22} {'A':>8} {'B':>8} {'Δ(A-B)':>10}  winner",
        "  " + "─" * 56,
    ]
    for m in report.metric_deltas:
        mark = "A" if m.winner == "A" else ("B" if m.winner == "B" else "=")
        lines.append(
            f"  {m.name:<22} {m.a_mean:>8.3f} {m.b_mean:>8.3f} {m.delta:>+10.4f}  {mark}"
        )
    lines.append("")
    a_flip = sum(1 for f in report.query_flips if f.winner == "A")
    b_flip = sum(1 for f in report.query_flips if f.winner == "B")
    lines.append(f"  per-query flips: {report.a_label} wins {a_flip}, {report.b_label} wins {b_flip} (of {report.n_queries})")
    return "\n".join(lines)
