"""Synthetic dataset generators — benchmark without a private corpus.

Each generator returns a `Dataset` of `Query` objects. The queries carry
`relevant_doc_ids` pointing into a companion corpus the generator also returns,
so a retriever can be built from the same corpus and graded fairly.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Optional

from ..types import Query, Dataset, RetrievedDoc


@dataclass(frozen=True)
class CorpusDoc:
    id: str
    text: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SyntheticDataset:
    """A dataset plus the corpus it draws relevance from."""

    dataset: Dataset
    corpus: tuple[CorpusDoc, ...]

    def as_retrieved_docs(self) -> dict[str, RetrievedDoc]:
        return {d.id: RetrievedDoc(id=d.id, text=d.text) for d in self.corpus}


# ── A tiny, deterministic factual corpus ────────────────────────────────────

_FACTS: list[CorpusDoc] = [
    CorpusDoc("d1", "The Eiffel Tower is located in Paris, France and was completed in 1889.", ("geography",)),
    CorpusDoc("d2", "Mount Everest is the highest mountain above sea level, peaking at 8,849 meters.", ("geography",)),
    CorpusDoc("d3", "Water boils at 100 degrees Celsius at standard atmospheric pressure.", ("science",)),
    CorpusDoc("d4", "The speed of light in a vacuum is approximately 299,792 kilometers per second.", ("science",)),
    CorpusDoc("d5", "Python was created by Guido van Rossum and first released in 1991.", ("computing",)),
    CorpusDoc("d6", "The Great Wall of China stretches over 21,000 kilometers in total length.", ("geography",)),
    CorpusDoc("d7", "Photosynthesis converts carbon dioxide and water into glucose and oxygen.", ("science",)),
    CorpusDoc("d8", "The TCP/IP protocol suite forms the foundation of the modern internet.", ("computing",)),
    CorpusDoc("d9", "The human heart has four chambers: two atria and two ventricles.", ("biology",)),
    CorpusDoc("d10", "Shakespeare wrote Hamlet around the year 1600.", ("literature",)),
]


def _build_factual_queries() -> list[Query]:
    return [
        Query("q1", "Where is the Eiffel Tower located?", "The Eiffel Tower is located in Paris, France.", ("d1",), {"category": "geography"}),
        Query("q2", "What is the tallest mountain above sea level?", "Mount Everest is the highest mountain above sea level.", ("d2",), {"category": "geography"}),
        Query("q3", "At what temperature does water boil at standard pressure?", "Water boils at 100 degrees Celsius at standard atmospheric pressure.", ("d3",), {"category": "science"}),
        Query("q4", "How fast does light travel in a vacuum?", "The speed of light in a vacuum is approximately 299,792 kilometers per second.", ("d4",), {"category": "science"}),
        Query("q5", "Who created the Python programming language?", "Python was created by Guido van Rossum.", ("d5",), {"category": "computing"}),
        Query("q6", "How long is the Great Wall of China?", "The Great Wall of China stretches over 21,000 kilometers.", ("d6",), {"category": "geography"}),
        Query("q7", "What does photosynthesis produce?", "Photosynthesis produces glucose and oxygen.", ("d7",), {"category": "science"}),
        Query("q8", "What protocol suite is the internet built on?", "The internet is built on the TCP/IP protocol suite.", ("d8",), {"category": "computing"}),
        Query("q9", "How many chambers does the human heart have?", "The human heart has four chambers.", ("d9",), {"category": "biology"}),
        Query("q10", "When was Hamlet written?", "Shakespeare wrote Hamlet around 1600.", ("d10",), {"category": "literature"}),
    ]


def factual(seed: Optional[int] = None) -> SyntheticDataset:
    """A 10-query factual dataset drawn from a 10-doc corpus.

    Deterministic by default; pass `seed` to shuffle query order for robustness checks.
    """
    queries = _build_factual_queries()
    if seed is not None:
        rng = random.Random(seed)
        queries = list(queries)
        rng.shuffle(queries)
    ds = Dataset(name="factual", version="0.1.0", queries=tuple(queries))
    return SyntheticDataset(dataset=ds, corpus=tuple(_FACTS))


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def multi_hop(n: int = 5) -> SyntheticDataset:
    """Queries whose answer requires combining two corpus docs.

    Each multi-hop query lists both docs as relevant, testing recall of
    the full evidence set rather than a single doc.
    """
    pairs = [
        (
            Query("mh1", "What do the tallest mountain and the Eiffel Tower have in common geographically?", "", ("d2", "d1"), {"category": "geography", "type": "multi-hop"}),
            "Mount Everest and the Eiffel Tower both relate to notable geography: one is the tallest mountain, the other a Paris landmark.",
        ),
        (
            Query("mh2", "Compare two scientific facts about light and water.", "", ("d4", "d3"), {"category": "science", "type": "multi-hop"}),
            "Light travels at ~299,792 km/s in a vacuum, while water boils at 100 C at standard pressure.",
        ),
    ]
    queries = [p[0] for p in pairs[:n]]
    ds = Dataset(name="multi-hop", version="0.1.0", queries=tuple(queries))
    return SyntheticDataset(dataset=ds, corpus=tuple(_FACTS))


# Re-export the adversarial dataset generator from the submodule.
from .adversarial import adversarial, adversarial_stats  # noqa: E402

__all__ = [
    "CorpusDoc",
    "SyntheticDataset",
    "factual",
    "multi_hop",
    "adversarial",
    "adversarial_stats",
]
