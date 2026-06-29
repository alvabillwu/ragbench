"""Adversarial synthetic dataset — stresses retrieval *precision*.

Each query has one ground-truth target doc plus several **hard-negative
distractors**: docs that share many surface keywords with the query/target but
answer a different (wrong) question. A naive keyword retriever ranks the
distractors highly, so precision@K drops — exactly the failure mode this
dataset is designed to surface.

Distractor patterns used:
  - near-miss number: same entity class, wrong figure (e.g. another mountain's height)
  - paraphrased-wrong: reuses the query's key phrase about a different subject
  - same-topic-different-fact: topically adjacent but not the asked fact
"""

from __future__ import annotations

from ..types import Query, Dataset
from . import CorpusDoc, SyntheticDataset


# Each entry: (target doc, [distractor docs], query_text, expected_answer, category)
_SPEC: list[tuple[CorpusDoc, list[CorpusDoc], str, str, str]] = [
    (
        CorpusDoc("adv-everest", "Mount Everest is the highest mountain above sea level, peaking at 8,849 meters on the Nepal-China border.", ("geography",)),
        [
            CorpusDoc("adv-kili", "Mount Kilimanjaro is the highest mountain in Africa, rising to 5,895 meters in Tanzania.", ("geography",)),
            CorpusDoc("adv-maunakea", "Mauna Kea is the tallest mountain measured from base to peak, rising over 10,000 meters from the ocean floor.", ("geography",)),
            CorpusDoc("adv-aconcagua", "Aconcagua is the highest mountain outside Asia, at 6,961 meters in the Andes.", ("geography",)),
        ],
        "What is the highest mountain above sea level?",
        "Mount Everest is the highest mountain above sea level at 8,849 meters.",
        "geography",
    ),
    (
        CorpusDoc("adv-light", "The speed of light in a vacuum is approximately 299,792 kilometers per second.", ("science",)),
        [
            CorpusDoc("adv-sound", "The speed of sound in air at 20 degrees Celsius is about 343 meters per second.", ("science",)),
            CorpusDoc("adv-light-water", "Light travels through water at roughly 225,000 kilometers per second, slower than in a vacuum.", ("science",)),
            CorpusDoc("adv-earth-orbit", "The Earth orbits the Sun at approximately 30 kilometers per second.", ("science",)),
        ],
        "How fast does light travel in a vacuum?",
        "Light travels at approximately 299,792 kilometers per second in a vacuum.",
        "science",
    ),
    (
        CorpusDoc("adv-python", "Python was created by Guido van Rossum and first released in 1991.", ("computing",)),
        [
            CorpusDoc("adv-ruby", "Ruby was created by Yukihiro Matsumoto and first released in 1995.", ("computing",)),
            CorpusDoc("adv-python3", "Python 3 was released in 2008, introducing backward-incompatible changes to the language.", ("computing",)),
            CorpusDoc("adv-java", "Java was created by James Gosling and first released in 1995.", ("computing",)),
        ],
        "Who created the Python programming language?",
        "Python was created by Guido van Rossum.",
        "computing",
    ),
    (
        CorpusDoc("adv-boiling", "Water boils at 100 degrees Celsius at standard atmospheric pressure.", ("science",)),
        [
            CorpusDoc("adv-freezing", "Water freezes at 0 degrees Celsius at standard atmospheric pressure.", ("science",)),
            CorpusDoc("adv-boiling-altitude", "Water boils below 100 degrees Celsius at high altitude due to lower atmospheric pressure.", ("science",)),
            CorpusDoc("adv-iron-melt", "Iron melts at approximately 1,538 degrees Celsius, far above water's boiling point.", ("science",)),
        ],
        "At what temperature does water boil at standard atmospheric pressure?",
        "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "science",
    ),
    (
        CorpusDoc("adv-eiffel", "The Eiffel Tower is located in Paris, France and was completed in 1889.", ("geography",)),
        [
            CorpusDoc("adv-statue", "The Statue of Liberty was dedicated in 1886 and stands in New York Harbor, USA.", ("geography",)),
            CorpusDoc("adv-tower-pisa", "The Leaning Tower of Pisa is located in Italy and was completed over several centuries.", ("geography",)),
            CorpusDoc("adv-paris-louvre", "The Louvre in Paris is the world's most-visited art museum, opened to the public in 1793.", ("geography",)),
        ],
        "Where is the Eiffel Tower located?",
        "The Eiffel Tower is located in Paris, France.",
        "geography",
    ),
]


def adversarial() -> SyntheticDataset:
    """A 5-query dataset where each query is padded with 3 hard-negative distractors.

    Designed to defeat keyword overlap: distractors share many query terms but
    answer a different question, so a good retriever must rank the single
    target doc above them.
    """
    corpus: list[CorpusDoc] = []
    queries: list[Query] = []
    for i, (target, distractors, qtext, expected, cat) in enumerate(_SPEC, start=1):
        corpus.append(target)
        corpus.extend(distractors)
        queries.append(
            Query(
                id=f"adv-{i}",
                text=qtext,
                expected_answer=expected,
                relevant_doc_ids=(target.id,),
                metadata={"category": cat, "type": "adversarial", "n_distractors": len(distractors)},
            )
        )
    ds = Dataset(name="adversarial", version="0.1.0", queries=tuple(queries))
    return SyntheticDataset(dataset=ds, corpus=tuple(corpus))


def adversarial_stats() -> dict:
    """Quick descriptor of the adversarial dataset's difficulty profile."""
    syn = adversarial()
    n_queries = len(syn.dataset)
    n_docs = len(syn.corpus)
    n_distractors = n_docs - n_queries  # one target per query
    return {
        "name": syn.dataset.name,
        "queries": n_queries,
        "corpus_docs": n_docs,
        "distractors": n_distractors,
        "distractors_per_query": n_distractors // n_queries if n_queries else 0,
        "precision_challenge": "high — distractors share query keywords but answer different questions",
    }
