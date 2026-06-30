"""Long-tail dataset — many rare, topically-sparse queries.

Real corpora have a long tail of uncommon questions. This dataset pairs a few
common queries with several long-tail queries (obscure facts, niche
terminology) so a benchmark can surface where retrieval falls off — typically
the tail, where keyword overlap is weaker and the right doc is one of many
sparse ones. Useful for stress-testing recall on low-signal queries.
"""

from __future__ import annotations

from ..types import Query, Dataset
from . import CorpusDoc, SyntheticDataset


# (doc, query_text, expected_answer, category, is_long_tail)
_SPEC: list[tuple[CorpusDoc, str, str, str, bool]] = [
    # ── head: common, high-signal queries ──
    (CorpusDoc("lt-eiffel", "The Eiffel Tower is located in Paris, France and was completed in 1889.", ("geography",)),
     "Where is the Eiffel Tower?", "The Eiffel Tower is located in Paris, France.", "geography", False),
    (CorpusDoc("lt-everest", "Mount Everest is the highest mountain above sea level, peaking at 8,849 meters.", ("geography",)),
     "What is the highest mountain above sea level?", "Mount Everest is the highest mountain above sea level.", "geography", False),
    # ── tail: rare, low-signal queries (the interesting part) ──
    (CorpusDoc("lt-voynich", "The Voynich manuscript is an illustrated codex from the early 15th century written in an undeciphered script.", ("history",)),
     "What is the Voynich manuscript?", "The Voynich manuscript is an undeciphered 15th-century illustrated codex.", "history", True),
    (CorpusDoc("lt-oklo", "The Oklo natural nuclear reactor in Gabon underwent spontaneous fission about 1.7 billion years ago.", ("science",)),
     "Where did a natural nuclear reactor occur?", "A natural nuclear reactor occurred at Oklo in Gabon.", "science", True),
    (CorpusDoc("lt-antikythera", "The Antikythera mechanism is an ancient Greek hand-powered orrery, dated to around 100 BCE, considered the first analog computer.", ("history",)),
     "What is the Antikythera mechanism?", "The Antikythera mechanism is an ancient Greek analog computer.", "history", True),
    (CorpusDoc("lt-tardigrade", "Tardigrades are microscopic animals that can survive extreme temperatures, radiation, and the vacuum of space.", ("biology",)),
     "What animal can survive the vacuum of space?", "Tardigrades can survive the vacuum of space.", "biology", True),
    (CorpusDoc("lt-quipu", "Quipu were Inca recording devices made of knotted cords, used for accounting and census data.", ("history",)),
     "How did the Inca record information?", "The Inca recorded information using knotted cords called quipu.", "history", True),
    (CorpusDoc("lt-pitch", "The pitch drop experiment in Brisbane has been slowly dripping bitumen since 1927, with fewer than ten drops observed.", ("science",)),
     "What is the pitch drop experiment?", "The pitch drop experiment is a long-running bitumen viscosity experiment started in 1927.", "science", True),
]


def longtail() -> SyntheticDataset:
    """An 8-query dataset mixing 2 head queries with 6 long-tail queries.

    Each query's `metadata.tail` flag lets analysis slice head vs tail
    performance — the tail is where retrieval typically degrades.
    """
    corpus: list[CorpusDoc] = []
    queries: list[Query] = []
    for i, (doc, qtext, expected, cat, is_tail) in enumerate(_SPEC, start=1):
        corpus.append(doc)
        queries.append(
            Query(
                id=f"lt-{i}",
                text=qtext,
                expected_answer=expected,
                relevant_doc_ids=(doc.id,),
                metadata={"category": cat, "type": "long-tail", "tail": is_tail},
            )
        )
    ds = Dataset(name="long-tail", version="0.1.0", queries=tuple(queries))
    return SyntheticDataset(dataset=ds, corpus=tuple(corpus))


def longtail_stats() -> dict:
    syn = longtail()
    head = sum(1 for q in syn.dataset if not q.metadata.get("tail"))
    tail = sum(1 for q in syn.dataset if q.metadata.get("tail"))
    return {
        "name": syn.dataset.name,
        "queries": len(syn.dataset),
        "corpus_docs": len(syn.corpus),
        "head_queries": head,
        "tail_queries": tail,
        "tail_ratio": round(tail / len(syn.dataset), 2),
    }
