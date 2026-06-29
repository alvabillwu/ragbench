"""CLI entry point for ragbench.

  ragbench run [--dataset factual|multi-hop] [--judge mock|llm] [--json]
  ragbench datasets
  ragbench metrics [--judge mock|llm]

The `run` command exercises a built-in reference pipeline (a keyword-overlap
retriever + an exact-answer generator) over a synthetic dataset, so you get a
scorecard without writing any code. Swap in your own pipeline via the library API.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from . import __version__
from .datasets import factual, multi_hop
from .types import RetrievedDoc, Answer
from .runner import run_benchmark, BenchmarkConfig, default_metrics, metrics_with_judge
from .judge import get_judge
from .report import scorecard, render_scorecard


def _utf8_stdout():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


# ── built-in reference pipeline (no user code needed) ───────────────────────


def _reference_retriever_factory(syn):
    docs_by_id = syn.as_retrieved_docs()

    def retrieve(query_text: str) -> list[RetrievedDoc]:
        # Rank docs by token overlap with the query (a real but simple baseline).
        qtokens = {t.lower() for t in query_text.replace("?", "").split()}
        scored = []
        for d in syn.corpus:
            dtokens = {t.lower() for t in d.text.replace(",", "").replace(".", "").split()}
            overlap = len(qtokens & dtokens)
            scored.append((overlap, docs_by_id[d.id]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored]

    return retrieve


def _reference_generator_factory(syn):
    def generate(query_text: str, docs: list[RetrievedDoc]) -> Answer:
        # Return the expected answer if the right doc was retrieved; else echo top doc.
        q = next((qq for qq in syn.dataset if qq.text == query_text), None)
        if q and q.expected_answer and any(d.id in q.relevant_doc_ids for d in docs):
            return Answer(text=q.expected_answer, cited_doc_ids=tuple(d.id for d in docs[:1]))
        if docs:
            return Answer(text=docs[0].text[:80], cited_doc_ids=(docs[0].id,))
        return Answer(text="")

    return generate


# ── commands ────────────────────────────────────────────────────────────────


def _get_dataset(name: str):
    if name == "factual":
        return factual()
    if name in ("multi-hop", "multihop", "multi_hop"):
        return multi_hop(n=2)
    raise SystemExit(f"unknown dataset: {name!r} (use 'factual' or 'multi-hop')")


def cmd_run(args) -> int:
    syn = _get_dataset(args.dataset)
    retriever = _reference_retriever_factory(syn)
    generator = _reference_generator_factory(syn)

    if args.judge == "llm":
        judge = get_judge("llm", model=args.model)
        metrics = metrics_with_judge(judge)
    else:
        metrics = metrics_with_judge() if args.judge == "mock" else default_metrics()

    cfg = BenchmarkConfig(metrics=metrics)
    run = run_benchmark(syn.dataset, retriever, generator, cfg)
    sc = scorecard(run)

    if args.json:
        out = sc.as_dict()
        out["dataset"] = syn.dataset.name
        print(json.dumps(out, indent=2))
    else:
        print(render_scorecard(sc))
    return 0


def cmd_datasets(args) -> int:
    for name, fn in [("factual", factual), ("multi-hop", lambda: multi_hop(n=2))]:
        syn = fn()
        print(f"{name:<12} {len(syn.dataset)} queries  corpus={len(syn.corpus)} docs  v{syn.dataset.version}")
    return 0


def cmd_metrics(args) -> int:
    if args.judge == "mock":
        metrics = metrics_with_judge()
    else:
        metrics = default_metrics()
    print(f"Metric suite ({len(metrics)} metrics):")
    for m in metrics:
        print(f"  {m.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragbench",
        description="Benchmark a RAG pipeline on synthetic datasets.",
    )
    parser.add_argument("--version", action="version", version=f"ragbench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the reference pipeline on a dataset and print a scorecard")
    p_run.add_argument("--dataset", default="factual", help="factual | multi-hop (default: factual)")
    p_run.add_argument("--judge", default="off", help="off | mock | llm (default: off)")
    p_run.add_argument("--model", default="gpt-4o-mini", help="model for llm judge")
    p_run.add_argument("--json", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_ds = sub.add_parser("datasets", help="List available datasets")
    p_ds.set_defaults(func=cmd_datasets)

    p_m = sub.add_parser("metrics", help="List the metrics in a suite")
    p_m.add_argument("--judge", default="off", help="off | mock | llm")
    p_m.set_defaults(func=cmd_metrics)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    _utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.func(args)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
