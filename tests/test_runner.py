"""End-to-end tests: runner + report + synthetic datasets."""

import pytest

from ragbench import run_benchmark, scorecard
from ragbench.datasets import factual, multi_hop
from ragbench.types import RetrievedDoc, Answer
from ragbench.runner import BenchmarkConfig, run_single
from ragbench.report import render_scorecard


class TestDatasets:
    def test_factual_has_10_queries(self):
        s = factual()
        assert len(s.dataset) == 10
        assert len(s.corpus) == 10

    def test_factual_queries_have_ground_truth(self):
        s = factual()
        for q in s.dataset:
            assert q.relevant_doc_ids, f"query {q.id} has no relevant docs"

    def test_factual_deterministic(self):
        a = factual()
        b = factual()
        assert [q.id for q in a.dataset] == [q.id for q in b.dataset]

    def test_factual_with_seed_shuffles(self):
        a = [q.id for q in factual().dataset]
        b = [q.id for q in factual(seed=42).dataset]
        assert a != b

    def test_multi_hop_multiple_relevant(self):
        s = multi_hop(n=2)
        for q in s.dataset:
            assert len(q.relevant_doc_ids) >= 2


class TestRunner:
    def setup_method(self):
        self.syn = factual()

    def _perfect_retriever(self, syn):
        docs_by_id = syn.as_retrieved_docs()

        def retrieve(query_text: str) -> list[RetrievedDoc]:
            # A "perfect" retriever returns exactly the relevant doc(s).
            q = next(qq for qq in syn.dataset if qq.text == query_text)
            return [docs_by_id[d] for d in q.relevant_doc_ids if d in docs_by_id]

        return retrieve

    def _echo_generator(self, syn):
        def generate(query_text: str, docs: list[RetrievedDoc]) -> Answer:
            q = next(qq for qq in syn.dataset if qq.text == query_text)
            return Answer(text=q.expected_answer, cited_doc_ids=tuple(d.id for d in docs[:1]))

        return generate

    def test_perfect_pipeline_scores_high(self):
        syn = self.syn
        run = run_benchmark(syn.dataset, self._perfect_retriever(syn), self._echo_generator(syn))
        sc = scorecard(run)
        assert sc.overall_mean > 0.9
        assert sc.metric_means["recall@k"] == pytest.approx(1.0)
        assert sc.metric_means["mrr"] == pytest.approx(1.0)

    def test_terrible_pipeline_scores_low(self):
        syn = self.syn

        def bad_retrieve(qt: str):
            docs = syn.as_retrieved_docs()
            # return all irrelevant first
            return [d for d in docs.values()][:1]

        def empty_gen(qt: str, docs):
            return Answer(text="")

        run = run_benchmark(syn.dataset, bad_retrieve, empty_gen)
        sc = scorecard(run)
        assert sc.overall_mean < 0.3

    def test_pipeline_exception_does_not_crash(self):
        syn = self.syn

        def explode(qt: str):
            raise RuntimeError("boom")

        def gen(qt: str, docs):
            return Answer("x")

        run = run_benchmark(syn.dataset, explode, gen)
        assert len(run.results) == len(syn.dataset)
        for r in run.results:
            assert r.retrieved == ()

    def test_run_single_overall_in_range(self):
        syn = self.syn
        q = syn.dataset.queries[0]
        ret = self._perfect_retriever(syn)
        gen = self._echo_generator(syn)
        r = run_single(q, ret, gen, BenchmarkConfig())
        assert 0.0 <= r.scores["overall"] <= 1.0

    def test_weights_normalize(self):
        # weights need not sum to 1; overall should still be in [0,1]
        cfg = BenchmarkConfig(weights={"mrr": 1.0, "recall@k": 1.0})
        syn = self.syn
        q = syn.dataset.queries[0]
        r = run_single(q, self._perfect_retriever(syn), self._echo_generator(syn), cfg)
        assert 0.0 <= r.scores["overall"] <= 1.0


class TestReport:
    def test_scorecard_dict_round(self):
        syn = factual()
        run = run_benchmark(
            syn.dataset,
            lambda qt: [syn.as_retrieved_docs()[i] for i in syn.dataset.queries[0].relevant_doc_ids],
            lambda qt, docs: Answer("answer"),
        )
        d = scorecard(run).as_dict()
        assert d["n_queries"] == 10
        assert isinstance(d["overall_mean"], float)

    def test_render_scorecard(self):
        syn = factual()
        run = run_benchmark(syn.dataset, lambda qt: [], lambda qt, d: Answer(""))
        out = render_scorecard(scorecard(run))
        assert "ragbench" in out
        assert "overall" in out
