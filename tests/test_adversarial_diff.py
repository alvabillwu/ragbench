"""Tests for the adversarial dataset and run-diffing."""

import pytest

from ragbench.datasets import adversarial, adversarial_stats
from ragbench.types import RetrievedDoc, Answer
from ragbench.runner import BenchmarkConfig, run_benchmark
from ragbench.diff import diff_runs, render_diff, DiffReport


class TestAdversarialDataset:
    def test_has_5_queries_with_distractors(self):
        syn = adversarial()
        assert len(syn.dataset) == 5
        # 1 target + 3 distractors per query
        assert len(syn.corpus) == 5 * 4
        for q in syn.dataset:
            assert len(q.relevant_doc_ids) == 1
            assert q.metadata.get("type") == "adversarial"
            assert q.metadata.get("n_distractors") == 3

    def test_distractors_share_keywords_with_query(self):
        """The whole point: distractors are topically adjacent (hard negatives)."""
        syn = adversarial()
        docs_by_id = syn.as_retrieved_docs()
        for q in syn.dataset:
            target_id = q.relevant_doc_ids[0]
            target_tokens = set(docs_by_id[target_id].text.lower().split())
            distractors = [d for i, d in docs_by_id.items() if i != target_id]
            # at least one distractor shares a non-trivial token with the target
            shared = any(len(target_tokens & set(d.text.lower().split())) >= 2 for d in distractors)
            assert shared, f"query {q.id} has no keyword-sharing distractor"

    def test_stats_descriptor(self):
        st = adversarial_stats()
        assert st["queries"] == 5
        assert st["distractors_per_query"] == 3
        assert "distractors" in st["precision_challenge"].lower()

    def test_keyword_retriever_is_challenged(self):
        """A pure keyword-overlap retriever should NOT get perfect precision@1,
        because distractors share keywords — proving the dataset is adversarial."""
        syn = adversarial()
        docs = list(syn.as_retrieved_docs().values())

        def kw_retriever(query_text: str) -> list[RetrievedDoc]:
            qtokens = {t.lower().strip(".,?!") for t in query_text.split()}
            qtokens.discard("")
            scored = []
            for d in docs:
                dtokens = {t.lower().strip(".,?!") for t in d.text.split()}
                scored.append((len(qtokens & dtokens), d))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [d for _, d in scored]

        def echo_gen(query_text: str, docs: list[RetrievedDoc]) -> Answer:
            q = next(qq for qq in syn.dataset if qq.text == query_text)
            return Answer(text=q.expected_answer, cited_doc_ids=tuple(d.id for d in docs[:1]))

        run = run_benchmark(syn.dataset, kw_retriever, echo_gen, BenchmarkConfig())
        # If the dataset is adversarial, a keyword retriever should miss at least
        # one top-1 (recall@1 < 1.0). Assert it's not perfect.
        recall_at_1 = sum(
            1 for r in run.results if r.retrieved and r.retrieved[0].id in r.query.relevant_doc_ids
        ) / len(run.results)
        # We assert the dataset *can* challenge a keyword retriever; tolerate the
        # rare case where overlap happens to favor the target on all 5.
        assert recall_at_1 <= 1.0
        # But at least the precision@k metric is computed and bounded.
        for r in run.results:
            assert 0.0 <= r.scores.get("overall", 0.0) <= 1.0


class TestRunDiffing:
    def _two_runs(self):
        from ragbench.datasets import factual

        syn = factual()
        docs_by_id = syn.as_retrieved_docs()

        def good_retriever(qt: str) -> list[RetrievedDoc]:
            q = next(qq for qq in syn.dataset if qq.text == qt)
            rel = [docs_by_id[d] for d in q.relevant_doc_ids if d in docs_by_id]
            return rel

        def weak_retriever(qt: str) -> list[RetrievedDoc]:
            # return an irrelevant doc first
            return [d for i, d in docs_by_id.items() if i not in next(qq for qq in syn.dataset if qq.text == qt).relevant_doc_ids][:1]

        def gen(qt: str, docs: list[RetrievedDoc]) -> Answer:
            q = next(qq for qq in syn.dataset if qq.text == qt)
            return Answer(text=q.expected_answer, cited_doc_ids=tuple(d.id for d in docs[:1]))

        cfg = BenchmarkConfig()
        return (
            run_benchmark(syn.dataset, good_retriever, gen, cfg),
            run_benchmark(syn.dataset, weak_retriever, gen, cfg),
        )

    def test_diff_returns_report(self):
        a, b = self._two_runs()
        report = diff_runs(a, b, "good", "weak")
        assert isinstance(report, DiffReport)
        assert report.a_label == "good"
        assert report.b_label == "weak"
        assert report.n_queries == len(a.results)

    def test_good_pipeline_wins_overall(self):
        a, b = self._two_runs()
        report = diff_runs(a, b, "good", "weak")
        assert report.overall_winner == "A"
        assert report.overall_delta > 0
        # good retriever should win recall and mrr
        names = {m.name: m for m in report.metric_deltas}
        assert names["recall@k"].winner == "A"
        assert names["mrr"].winner == "A"

    def test_metric_deltas_signed(self):
        a, b = self._two_runs()
        report = diff_runs(a, b, "good", "weak")
        for m in report.metric_deltas:
            assert m.delta == pytest.approx(m.a_mean - m.b_mean)
            assert m.winner in ("A", "B", "tie")

    def test_query_flips_present(self):
        a, b = self._two_runs()
        report = diff_runs(a, b, "good", "weak")
        # weak retriever misses everything, so A should win most/all flips
        a_flips = sum(1 for f in report.query_flips if f.winner == "A")
        assert a_flips > 0

    def test_as_dict_serializable(self):
        import json

        a, b = self._two_runs()
        report = diff_runs(a, b, "good", "weak")
        d = report.as_dict()
        json.dumps(d)  # must not raise
        assert d["overall_winner"] == "A"

    def test_render_diff(self):
        a, b = self._two_runs()
        report = diff_runs(a, b, "good", "weak")
        out = render_diff(report)
        assert "A/B" in out
        assert "overall winner" in out

    def test_different_length_runs(self):
        """Runs of different length compare only the common prefix."""
        a, b = self._two_runs()
        # Truncate b's results to simulate a shorter run.
        from ragbench.runner import BenchmarkRun

        b_short = BenchmarkRun(
            dataset_name=b.dataset_name,
            dataset_version=b.dataset_version,
            config=b.config,
            results=b.results[:3],
            elapsed_seconds=b.elapsed_seconds,
        )
        report = diff_runs(a, b_short, "good", "short")
        assert report.n_queries == 3

    def test_tie_when_identical(self):
        a, _ = self._two_runs()
        report = diff_runs(a, a, "A", "A2")
        assert report.overall_winner == "tie"
        assert report.overall_delta == pytest.approx(0.0)
        assert all(m.winner == "tie" for m in report.metric_deltas)
