"""Tests for the judge backend and judge-based metrics."""

import pytest

from ragbench.types import Query, RetrievedDoc, Answer
from ragbench.judge import MockJudge, get_judge, _parse_verdict, Verdict
from ragbench.metrics.judge_metrics import FaithfulnessJudge, AnswerRelevanceJudge


class TestMockJudge:
    def setup_method(self):
        self.judge = MockJudge()

    def test_faithful_answer_scores_high(self):
        q = Query("q", "What is photosynthesis?")
        ctx = [RetrievedDoc(id="d1", text="Photosynthesis produces glucose and oxygen from water and CO2.")]
        a = Answer("glucose and oxygen")
        v = self.judge.judge("Score faithfulness — is the answer grounded in context?", q, ctx, a)
        assert v.score == pytest.approx(1.0)
        assert "grounded" in v.rationale

    def test_ungrounded_answer_scores_low(self):
        q = Query("q", "What is photosynthesis?")
        ctx = [RetrievedDoc(id="d1", text="The tower is in Paris.")]
        a = Answer("glucose and oxygen")  # tokens not in context
        v = self.judge.judge("Score faithfulness", q, ctx, a)
        assert v.score < 0.5

    def test_relevance_addresses_question(self):
        q = Query("q", "Where is the Eiffel Tower located")
        ctx = [RetrievedDoc(id="d1", text="The Eiffel Tower is in Paris.")]
        a = Answer("The Eiffel Tower is located in Paris")
        v = self.judge.judge("Score answer relevance to the question", q, ctx, a)
        # query tokens (eiffel, tower, located) all in answer
        assert v.score > 0.5

    def test_empty_answer(self):
        q = Query("q", "q")
        v = self.judge.judge("faithfulness", q, [RetrievedDoc(id="d", text="x")], Answer(""))
        assert v.score == 0.0

    def test_deterministic(self):
        q = Query("q", "q")
        ctx = [RetrievedDoc(id="d", text="the quick brown fox")]
        a = Answer("quick brown fox")
        v1 = self.judge.judge("faithfulness", q, ctx, a)
        v2 = self.judge.judge("faithfulness", q, ctx, a)
        assert v1 == v2


class TestGetJudge:
    def test_mock_factory(self):
        assert isinstance(get_judge("mock"), MockJudge)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            get_judge("bogus")


class TestParseVerdict:
    def test_parses_clean_json(self):
        v = _parse_verdict('{"score": 0.8, "rationale": "good"}')
        assert v.score == pytest.approx(0.8)
        assert v.rationale == "good"

    def test_parses_json_embedded_in_text(self):
        v = _parse_verdict('Sure! {"score": 0.5, "rationale": "ok"} hope that helps')
        assert v.score == 0.5

    def test_clamps_out_of_range(self):
        v = _parse_verdict('{"score": 1.7, "rationale": ""}')
        assert v.score == 1.0

    def test_unparseable_returns_zero(self):
        v = _parse_verdict("no json here at all")
        assert v.score == 0.0


class TestJudgeMetrics:
    def test_faithfulness_metric_protocol(self):
        m = FaithfulnessJudge()
        assert m.name == "faithfulness_judge"
        q = Query("q", "q")
        ctx = [RetrievedDoc(id="d", text="cats and dogs")]
        r = m.score(q, ctx, Answer("cats dogs"))
        assert 0.0 <= r.value <= 1.0
        assert "rationale" in r.detail

    def test_relevance_metric_protocol(self):
        m = AnswerRelevanceJudge()
        assert m.name == "answer_relevance_judge"
        q = Query("q", "capital of france")
        r = m.score(q, [RetrievedDoc(id="d", text="x")], Answer("the capital of france is paris"))
        assert 0.0 <= r.value <= 1.0

    def test_empty_answer_short_circuits(self):
        m = FaithfulnessJudge()
        r = m.score(Query("q", "q"), [RetrievedDoc(id="d", text="x")], Answer(""))
        assert r.value == 0.0
        assert r.detail["reason"] == "empty answer"

    def test_end_to_end_through_config(self):
        from ragbench.runner import BenchmarkConfig, run_single

        cfg = BenchmarkConfig()  # uses default metrics, no judge
        q = Query("q1", "Where is the Eiffel Tower?", "The Eiffel Tower is located in Paris, France.", ("d1",))
        retrieved = [RetrievedDoc(id="d1", text="The Eiffel Tower is located in Paris, France.")]
        answer = Answer("The Eiffel Tower is located in Paris, France.")
        r = run_single(q, lambda qt: retrieved, lambda qt, d: answer, cfg)
        assert 0.0 <= r.scores["overall"] <= 1.0
