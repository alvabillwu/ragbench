"""Tests for retrieval + generation metrics."""

import pytest

from ragbench.types import Query, RetrievedDoc, Answer
from ragbench.metrics import PrecisionAtK, RecallAtK, MRR, NDCGAtK
from ragbench.metrics.generation import AnswerOverlap, FaithfulnessLite, CitationCoverage


def _docs(*ids: str) -> list[RetrievedDoc]:
    return [RetrievedDoc(id=i, text=f"text {i}", score=float(j)) for j, i in enumerate(ids)]


class TestRetrievalMetrics:
    def setup_method(self):
        self.q = Query("q1", "q", relevant_doc_ids=("d1", "d2"))

    def test_precision_at_k(self):
        m = PrecisionAtK(k=3)
        # top-3 = [d1, dX, d2] -> 2 relevant of 3
        r = m.score(self.q, _docs("d1", "dX", "d2"), Answer(""))
        assert r.value == pytest.approx(2 / 3)

    def test_precision_no_relevant_returns_zero(self):
        m = PrecisionAtK(k=3)
        assert m.score(Query("q", "q"), _docs("d1"), Answer("")).value == 0.0

    def test_recall_at_k(self):
        m = RecallAtK(k=5)
        # relevant = {d1,d2}, top-5 has d1 -> 1 of 2
        r = m.score(self.q, _docs("dX", "dY", "d1", "dZ", "dW"), Answer(""))
        assert r.value == pytest.approx(0.5)

    def test_recall_all_retrieved(self):
        m = RecallAtK(k=5)
        r = m.score(self.q, _docs("d1", "d2"), Answer(""))
        assert r.value == 1.0

    def test_mrr_first_hit_rank1(self):
        r = MRR().score(self.q, _docs("d1", "d2"), Answer(""))
        assert r.value == 1.0

    def test_mrr_first_hit_rank3(self):
        r = MRR().score(self.q, _docs("dX", "dY", "d1"), Answer(""))
        assert r.value == pytest.approx(1 / 3)

    def test_mrr_no_hit(self):
        assert MRR().score(self.q, _docs("dX", "dY"), Answer("")).value == 0.0

    def test_ndcg_perfect_ranking(self):
        r = NDCGAtK(k=5).score(self.q, _docs("d1", "d2"), Answer(""))
        assert r.value == pytest.approx(1.0)

    def test_ndcg_suboptimal(self):
        # d1 at rank 2, d2 absent -> dcg = 1/log2(3) ; ideal = 1 + 1/log2(3)
        r = NDCGAtK(k=5).score(self.q, _docs("dX", "d1"), Answer(""))
        assert 0.0 < r.value < 1.0

    def test_ndcg_no_relevant(self):
        assert NDCGAtK(k=5).score(Query("q", "q"), _docs("d1"), Answer("")).value == 0.0


class TestGenerationMetrics:
    def test_answer_overlap_identical(self):
        q = Query("q", "q", expected_answer="the cat sat on the mat")
        a = Answer("the cat sat on the mat")
        assert AnswerOverlap().score(q, [], a).value == pytest.approx(1.0)

    def test_answer_overlap_partial(self):
        q = Query("q", "q", expected_answer="the cat sat on the mat")
        a = Answer("the cat ran fast")
        v = AnswerOverlap().score(q, [], a).value
        assert 0.0 < v < 1.0

    def test_answer_overlap_no_expected(self):
        q = Query("q", "q")
        assert AnswerOverlap().score(q, [], Answer("anything")).value == 0.0

    def test_faithfulness_all_supported(self):
        q = Query("q", "q")
        docs = [RetrievedDoc(id="d1", text="photosynthesis produces glucose and oxygen")]
        a = Answer("glucose and oxygen")
        assert FaithfulnessLite().score(q, docs, a).value == pytest.approx(1.0)

    def test_faithfulness_invents_content(self):
        q = Query("q", "q")
        docs = [RetrievedDoc(id="d1", text="the tower is in paris")]
        a = Answer("the tower is in tokyo")  # tokyo not in context
        v = FaithfulnessLite().score(q, docs, a).value
        assert v < 1.0

    def test_citation_coverage_valid(self):
        docs = _docs("d1", "d2")
        a = Answer("x", cited_doc_ids=("d1", "d2"))
        assert CitationCoverage().score(Query("q", "q"), docs, a).value == 1.0

    def test_citation_coverage_hallucinated(self):
        docs = _docs("d1")
        a = Answer("x", cited_doc_ids=("d1", "d999"))
        assert CitationCoverage().score(Query("q", "q"), docs, a).value == 0.5

    def test_citation_coverage_none(self):
        assert CitationCoverage().score(Query("q", "q"), [], Answer("x")).value == 0.0
