#!/usr/bin/env python3
"""ragbench example — benchmark a toy retriever+generator on the factual dataset.

Run:  python examples/benchmark.py
"""

from ragbench import run_benchmark, scorecard
from ragbench.datasets import factual
from ragbench.report import render_scorecard
from ragbench.types import RetrievedDoc, Answer


def main() -> None:
    syn = factual()
    docs_by_id = syn.as_retrieved_docs()

    def retriever(query_text: str) -> list[RetrievedDoc]:
        # A "perfect" toy retriever: return the relevant doc first.
        q = next(qq for qq in syn.dataset if qq.text == query_text)
        rel = [docs_by_id[d] for d in q.relevant_doc_ids if d in docs_by_id]
        rest = [d for i, d in docs_by_id.items() if i not in q.relevant_doc_ids]
        return rel + rest

    def generator(query_text: str, docs: list[RetrievedDoc]) -> Answer:
        q = next(qq for qq in syn.dataset if qq.text == query_text)
        return Answer(text=q.expected_answer, cited_doc_ids=tuple(d.id for d in docs[:1]))

    run = run_benchmark(syn.dataset, retriever, generator)
    print(render_scorecard(scorecard(run)))
    print()
    print("Per-query (first 3):")
    for r in run.results[:3]:
        print(f"  {r.query.id}  overall={r.scores['overall']:.3f}  mrr={r.scores.get('mrr', 0):.2f}")


if __name__ == "__main__":
    main()
