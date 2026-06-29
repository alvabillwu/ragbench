"""LLM-as-judge backends.

The judge scores a proposition (e.g. "is this answer faithful to the context?")
and returns a 0..1 score plus a short rationale. The backend is pluggable:

  - MockJudge: deterministic, no network — for tests and offline demos. Scores
    via cheap heuristics so it's a *real* baseline, not a random stub.
  - LLMJudge: optional, uses the `openai` package if installed. Requires an
    API key at runtime; not imported unless requested.

Keeping the judge behind a Protocol means metrics (faithfulness, relevance)
work identically whether you use the mock or a real LLM — swap the backend,
keep the metric.
"""

from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass
from typing import Optional, Protocol

from .types import Query, RetrievedDoc, Answer


@dataclass(frozen=True)
class Verdict:
    """A judge's verdict on one proposition."""

    score: float  # 0..1
    rationale: str = ""


class JudgeBackend(Protocol):
    """A backend that scores a proposition about (query, context, answer)."""

    name: str

    def judge(
        self,
        instruction: str,
        query: Query,
        context: list[RetrievedDoc],
        answer: Answer,
    ) -> Verdict: ...


# ── MockJudge: deterministic heuristic baseline ─────────────────────────────

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text)}


class MockJudge:
    """A no-network judge that scores via token overlap heuristics.

    Not a real LLM, but a useful deterministic baseline: faithfulness = fraction
    of answer tokens in context; relevance = fraction of query tokens in answer.
    Identical inputs always produce identical verdicts — perfect for tests.
    """

    name = "mock"

    def judge(
        self,
        instruction: str,
        query: Query,
        context: list[RetrievedDoc],
        answer: Answer,
    ) -> Verdict:
        inst = instruction.lower()
        ans_tokens = _tokens(answer.text)
        if not ans_tokens:
            return Verdict(0.0, "empty answer")

        if "faithful" in inst or "faithfulness" in inst or "grounded" in inst:
            ctx_tokens: set[str] = set()
            for d in context:
                ctx_tokens |= _tokens(d.text)
            if not ctx_tokens:
                return Verdict(0.0, "no context to ground in")
            supported = len(ans_tokens & ctx_tokens)
            score = supported / len(ans_tokens)
            return Verdict(score, f"{supported}/{len(ans_tokens)} answer tokens grounded in context")

        if "relev" in inst:  # relevant / relevance
            q_tokens = _tokens(query.text)
            if not q_tokens:
                return Verdict(0.0, "no query tokens")
            hit = len(q_tokens & ans_tokens)
            score = min(1.0, hit / max(1, len(q_tokens)))
            return Verdict(score, f"{hit}/{len(q_tokens)} query tokens addressed")

        # Generic fallback: overlap of answer with (query + context).
        pool = _tokens(query.text)
        for d in context:
            pool |= _tokens(d.text)
        score = len(ans_tokens & pool) / len(ans_tokens) if ans_tokens else 0.0
        return Verdict(score, "answer/context+query overlap")


# ── LLMJudge: optional real backend ─────────────────────────────────────────

_DEFAULT_PROMPT = """You are an evaluator. {instruction}

Question: {question}
Context: {context}
Answer: {answer}

Respond with ONLY a JSON object: {{"score": <float 0-1>, "rationale": "<short>"}}"""


class LLMJudge:
    """A judge backed by an OpenAI-compatible chat model.

    Lazy-imports `openai` only when used, so the package is an optional
    dependency. Requires OPENAI_API_KEY (or a provided client) at judge time.
    """

    name = "llm"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        client=None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self._client = client
        self._api_key = api_key
        self._base_url = base_url

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            openai = importlib.import_module("openai")  # type: ignore
        except ImportError as e:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "LLMJudge requires the 'openai' package. Install with: pip install 'ragbench[judge]'"
            ) from e
        key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not key:  # pragma: no cover - env-dependent
            raise RuntimeError("OPENAI_API_KEY is required for LLMJudge")
        kwargs = {"api_key": key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = openai.OpenAI(**kwargs)  # type: ignore[attr-defined]
        return self._client

    def judge(
        self,
        instruction: str,
        query: Query,
        context: list[RetrievedDoc],
        answer: Answer,
    ) -> Verdict:  # pragma: no cover - network call
        client = self._get_client()
        ctx_text = "\n".join(d.text for d in context) or "(none)"
        prompt = _DEFAULT_PROMPT.format(
            instruction=instruction,
            question=query.text,
            context=ctx_text,
            answer=answer.text,
        )
        resp = client.chat.completions.create(  # type: ignore[union-attr]
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        return _parse_verdict(text)


def _parse_verdict(text: str) -> Verdict:
    """Parse the model's JSON response into a Verdict, tolerating noise."""
    # Find the first {...} block.
    m = re.search(r"\{[^{}]*\}", text)
    if not m:
        # Fallback: if the model didn't emit JSON, score 0 with raw text.
        return Verdict(0.0, f"unparseable: {text[:120]}")
    import json

    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return Verdict(0.0, f"unparseable: {text[:120]}")
    score = float(d.get("score", 0.0))
    score = max(0.0, min(1.0, score))
    return Verdict(score, str(d.get("rationale", "")))


def get_judge(kind: str = "mock", **kwargs) -> JudgeBackend:
    """Factory: 'mock' -> MockJudge, 'llm' -> LLMJudge."""
    kind = kind.lower()
    if kind == "mock":
        return MockJudge()
    if kind == "llm":
        return LLMJudge(**kwargs)
    raise ValueError(f"unknown judge kind: {kind!r} (use 'mock' or 'llm')")
