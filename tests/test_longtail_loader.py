"""Tests for the long-tail dataset and the pluggable pipeline loader."""

import textwrap

import pytest

from ragbench.datasets import longtail, longtail_stats
from ragbench.types import RetrievedDoc, Answer
from ragbench.runner import BenchmarkConfig, run_benchmark
from ragbench.pipeline_loader import load_pipeline, PipelineLoadError


class TestLongTailDataset:
    def test_has_8_queries_with_head_and_tail(self):
        syn = longtail()
        assert len(syn.dataset) == 8
        assert len(syn.corpus) == 8
        head = [q for q in syn.dataset if not q.metadata.get("tail")]
        tail = [q for q in syn.dataset if q.metadata.get("tail")]
        assert len(head) == 2
        assert len(tail) == 6

    def test_queries_have_ground_truth(self):
        syn = longtail()
        for q in syn.dataset:
            assert len(q.relevant_doc_ids) == 1
            assert q.expected_answer
            assert q.metadata.get("type") == "long-tail"

    def test_stats(self):
        st = longtail_stats()
        assert st["queries"] == 8
        assert st["head_queries"] == 2
        assert st["tail_queries"] == 6
        assert st["tail_ratio"] == 0.75

    def test_runs_through_benchmark(self):
        syn = longtail()
        docs_by_id = syn.as_retrieved_docs()

        def retriever(qt: str) -> list[RetrievedDoc]:
            q = next(qq for qq in syn.dataset if qq.text == qt)
            return [docs_by_id[d] for d in q.relevant_doc_ids if d in docs_by_id]

        def gen(qt: str, docs: list[RetrievedDoc]) -> Answer:
            q = next(qq for qq in syn.dataset if qq.text == qt)
            return Answer(text=q.expected_answer, cited_doc_ids=tuple(d.id for d in docs[:1]))

        run = run_benchmark(syn.dataset, retriever, gen, BenchmarkConfig())
        assert len(run.results) == 8
        # tail queries are reachable and scored
        tail_results = [r for r in run.results if r.query.metadata.get("tail")]
        assert len(tail_results) == 6


class TestPipelineLoader:
    def _write_pipeline(self, tmp_path, body: str):
        p = tmp_path / "mypipe.py"
        p.write_text(textwrap.dedent(body), encoding="utf-8")
        return p

    def test_loads_retriever_and_generator(self, tmp_path):
        p = self._write_pipeline(
            tmp_path,
            """
            from ragbench.types import RetrievedDoc, Answer
            def retriever(query):
                return [RetrievedDoc(id="d1", text="hello")]
            def generator(query, docs):
                return Answer(text="ok")
            """,
        )
        retriever, generator = load_pipeline(p)
        assert retriever("q") == [RetrievedDoc(id="d1", text="hello")]
        assert generator("q", []).text == "ok"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(PipelineLoadError, match="not found"):
            load_pipeline(tmp_path / "nonexistent.py")

    def test_non_python_file_raises(self, tmp_path):
        p = tmp_path / "pipe.txt"
        p.write_text("x", encoding="utf-8")
        with pytest.raises(PipelineLoadError, match="must be a .py"):
            load_pipeline(p)

    def test_missing_exports_raises(self, tmp_path):
        p = self._write_pipeline(tmp_path, "x = 1\n")
        with pytest.raises(PipelineLoadError, match="missing required export"):
            load_pipeline(p)

    def test_non_callable_export_raises(self, tmp_path):
        p = self._write_pipeline(
            tmp_path,
            """
            retriever = 5
            generator = "not a function"
            """,
        )
        with pytest.raises(PipelineLoadError, match="not callable"):
            load_pipeline(p)

    def test_import_error_in_pipeline_raises(self, tmp_path):
        p = self._write_pipeline(
            tmp_path,
            """
            import nonexistent_module_xyz
            def retriever(q): return []
            def generator(q, d): return None
            """,
        )
        with pytest.raises(PipelineLoadError, match="error importing"):
            load_pipeline(p)

    def test_end_to_end_with_benchmark(self, tmp_path):
        """A user pipeline loaded from a file runs through run_benchmark."""
        from ragbench.datasets import factual

        syn = factual()
        docs_by_id = syn.as_retrieved_docs()
        p = self._write_pipeline(
            tmp_path,
            f"""
            from ragbench.types import RetrievedDoc, Answer
            _docs = {dict((k, v.to_dict() if hasattr(v, 'to_dict') else {'id': v.id, 'text': v.text, 'score': v.score}) for k, v in docs_by_id.items())}
            _relevance = {dict((q.id, list(q.relevant_doc_ids)) for q in syn.dataset)}
            _queries = {dict((q.id, q.text) for q in syn.dataset)}
            _answers = {dict((q.id, q.expected_answer) for q in syn.dataset)}
            def retriever(query):
                for qid, qt in _queries.items():
                    if qt == query:
                        return [RetrievedDoc(id=d, text=_docs[d]['text']) for d in _relevance[qid] if d in _docs]
                return []
            def generator(query, docs):
                for qid, qt in _queries.items():
                    if qt == query:
                        return Answer(text=_answers[qid], cited_doc_ids=tuple(d.id for d in docs[:1]))
                return Answer(text="")
            """,
        )
        retriever, generator = load_pipeline(p)
        run = run_benchmark(syn.dataset, retriever, generator, BenchmarkConfig())
        assert len(run.results) == 10
        # perfect retriever → high overall
        from ragbench.report import scorecard

        assert scorecard(run).overall_mean > 0.8
