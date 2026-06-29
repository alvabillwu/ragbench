# 📊 ragbench

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange)](#)

**Lightweight RAG pipeline benchmarking framework.**

Benchmark any retrieval-augmented-generation pipeline with synthetic datasets and deterministic metrics — **no LLM API required for the core**. Plug in your retriever and generator (LangChain, LlamaIndex, or hand-rolled), run over a built-in dataset, and get a reproducible scorecard of retrieval and generation quality.

> Built incrementally as a medium-complexity project. Retrieval + generation metrics and synthetic datasets ship first; an optional LLM-as-judge backend is planned.

## Features

- 🧪 **Synthetic datasets** — a 10-query factual set + multi-hop set, with ground-truth relevance, so you can benchmark without a private corpus
- 📏 **Deterministic metrics** — Precision@K, Recall@K, MRR, NDCG@K (retrieval) + AnswerOverlap, Faithfulness-lite, CitationCoverage (generation)
- 🔌 **Framework-agnostic** — your pipeline is just `(retriever, generator)` callables
- 📋 **Scorecards** — per-metric means ± stdev, overall weighted score, pass-rate
- 🔁 **Reproducible** — versioned datasets, no network calls in the core
- 🚫 **Zero hard dependencies** — pure stdlib; optional `openai` for the planned judge backend

## Quick Start

```bash
pip install ragbench
```

## Usage

### Benchmark a pipeline

```python
from ragbench import run_benchmark, scorecard
from ragbench.datasets import factual
from ragbench.report import render_scorecard
from ragbench.types import RetrievedDoc, Answer

syn = factual()                       # dataset + companion corpus
docs = syn.as_retrieved_docs()

def retriever(query: str) -> list[RetrievedDoc]:
    ...  # your retrieval logic, returns RetrievedDoc lists

def generator(query: str, docs: list[RetrievedDoc]) -> Answer:
    ...  # your LLM call, returns an Answer

run = run_benchmark(syn.dataset, retriever, generator)
print(render_scorecard(scorecard(run)))
```

Example output:
```
ragbench — factual (10 queries, 0.01s)
  overall: 0.940   pass-rate: 100.0%
  metrics:
    answer_overlap        1.000 ±0.000  ████████████████████
    citation_coverage     1.000 ±0.000  ████████████████████
    faithfulness_lite     0.980 ±0.040  ███████████████████
    mrr                   1.000 ±0.000  ████████████████████
    ndcg@k                1.000 ±0.000  ████████████████████
    precision@k           0.200 ±0.000  ████
    recall@k              1.000 ±0.000  ████████████████████
```

### Datasets

| Dataset | Queries | Notes |
|---------|---------|-------|
| `factual()` | 10 | single-doc factual Q&A across 5 categories |
| `multi_hop(n)` | n | two-doc evidence, tests full-evidence recall |

## Metrics

**Retrieval** (need `relevant_doc_ids` ground truth): Precision@K, Recall@K, MRR, NDCG@K.

**Generation** (no LLM judge): AnswerOverlap (token Jaccard vs expected), Faithfulness-lite (fraction of answer tokens supported by context — a hallucination proxy), CitationCoverage (cited docs that were actually retrieved).

## Roadmap

- [ ] LLM-as-judge backend (faithfulness, answer-relevance) — optional `openai` dep
- [ ] CLI: `ragbench run <pipeline_module> --dataset factual`
- [ ] Run diffing: compare two pipelines head-to-head
- [ ] More synthetic datasets (adversarial, long-tail)

## Development

```bash
git clone https://github.com/alvabillwu/ragbench.git
cd ragbench
pip install -e ".[dev]"
pytest -v
python examples/benchmark.py
```

## License

MIT © [alvabillwu](https://github.com/alvabillwu)
