"""Hybrid retrieval with a deterministic fake embedder, and the evaluation's scoring rules."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from chavruta.corpus import Chunk
from chavruta.embeddings import EMBEDDING_DIM, Vectors, load_vectors, save_vectors
from chavruta.evaluation import Question, Ranking, load_questions, score_rankings, verify_questions
from chavruta.retrieval import Retriever, matches
from chavruta.text import tokens


class HashEmbedder:
    """Bag-of-tokens vectors: texts sharing words are close. Deterministic, no model download."""

    key = "hash"
    dim = EMBEDDING_DIM

    def _one(self, text: str) -> Vectors:
        v = np.zeros(self.dim, dtype=np.float32)
        for t in tokens(text):
            v[int(hashlib.sha256(t.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def queries(self, texts: list[str]) -> Vectors:
        return np.stack([self._one(t) for t in texts])

    def passages(self, texts: list[str]) -> Vectors:
        return self.queries(texts)


def chunk(ref: str, text: str, language: str = "he") -> Chunk:
    return Chunk(ref, ref.rsplit(" ", 1)[0], "Test", language, "v", "Public Domain", text, ref, 0)


CHUNKS = [
    chunk("Deuteronomy 6:4", "שמע ישראל יהוה אלהינו יהוה אחד"),
    chunk("Deuteronomy 6:4", "Hear, O Israel: the LORD our God, the LORD is one.", "en"),
    chunk("Deuteronomy 6:7", "ושננתם לבניך ודברת בם בשבתך בביתך ובלכתך בדרך ובשכבך ובקומך"),
    chunk("Mishnah Berakhot 1:1", "מאימתי קורין את שמע בערבית משעה שהכהנים נכנסים לאכול בתרומתן"),
    chunk("Mishnah Berakhot 1:1", "From when may one recite the Shema in the evening?", "en"),
    chunk("Genesis 1:1", "בראשית ברא אלהים את השמים ואת הארץ"),
    chunk("Genesis 1:10", "ויקרא אלהים ליבשה ארץ"),
]


@pytest.fixture
def retriever() -> Retriever:
    e = HashEmbedder()
    return Retriever(CHUNKS, e, e.passages([c.text for c in CHUNKS]))


def test_keyword_search_finds_exact_hebrew_phrases(retriever: Retriever) -> None:
    hits = retriever.search("מאימתי קורין את שמע", k=3, mode="keyword")
    assert hits[0].chunk.ref == "Mishnah Berakhot 1:1"


def test_prefix_variants_match_unprefixed_query(retriever: Retriever) -> None:
    hits = retriever.search("בשכבך", k=1, mode="keyword")
    assert hits[0].chunk.ref == "Deuteronomy 6:7"


def test_results_are_distinct_references(retriever: Retriever) -> None:
    hits = retriever.search("שמע Shema Israel", k=10)
    refs = [h.chunk.ref for h in hits]
    assert len(refs) == len(set(refs))
    assert "Deuteronomy 6:4" in refs


def test_semantic_side_reaches_english_chunks(retriever: Retriever) -> None:
    hits = retriever.search("recite the Shema evening", k=2, mode="semantic")
    assert hits[0].chunk.ref == "Mishnah Berakhot 1:1"


def test_reranker_reorders_candidates(retriever: Retriever) -> None:
    class PreferGenesis:
        def scores(self, query: str, passages: list[str]) -> list[float]:
            return [1.0 if "בראשית" in p else 0.0 for p in passages]

    retriever.reranker = PreferGenesis()
    hits = retriever.search("אלהים ארץ", k=2, rerank=True)
    assert hits[0].chunk.ref == "Genesis 1:1"
    no_rerank = retriever.search("אלהים ארץ", k=2, rerank=False)
    assert {h.chunk.ref for h in no_rerank} == {"Genesis 1:1", "Genesis 1:10"}


def test_empty_or_punctuation_query_returns_nothing_from_keywords(retriever: Retriever) -> None:
    assert retriever.keyword("?!") == []


def test_vectors_must_match_chunks() -> None:
    with pytest.raises(ValueError, match="one vector per chunk"):
        Retriever(CHUNKS, HashEmbedder(), np.zeros((2, EMBEDDING_DIM), dtype=np.float32))


def test_vectors_roundtrip(tmp_path: Path) -> None:
    v = HashEmbedder().passages(["שמע", "ישראל"])
    save_vectors(tmp_path, ["a", "b"], v)
    ids, loaded = load_vectors(tmp_path)
    assert ids == ["a", "b"]
    assert np.allclose(loaded, v, atol=1e-3)


@pytest.mark.parametrize(
    ("retrieved", "gold", "ok"),
    [
        ("Berakhot 2a:3", "Berakhot 2a", True),
        ("Berakhot 2a", "Berakhot 2a", True),
        ("Genesis 1:10", "Genesis 1:1", False),
        ("Genesis 1:1", "Genesis 1:10", False),
        ("Rashi on Berakhot 2a:1:1", "Berakhot 2a", False),
        ("Berakhot 2a", "Berakhot 2a:3", False),
    ],
)
def test_matches_same_or_finer_reference(retrieved: str, gold: str, ok: bool) -> None:
    assert matches(retrieved, gold) is ok


# ---- evaluation scoring ---------------------------------------------------------------------------


def q(qid: str, gold: tuple[str, ...], lang: str = "en", typ: str = "source") -> Question:
    return Question(qid, "dev", lang, typ, "?", gold, {})


def test_recall_counts_any_gold_alternative_and_mrr_uses_the_first() -> None:
    questions = [q("a", ("Berakhot 2a", "Mishnah Berakhot 1:1")), q("b", ("Genesis 1:1",), "he")]
    rankings = {
        "a": Ranking("a", ("x", "y", "Mishnah Berakhot 1:1", *["z"] * 7), 0.01),
        "b": Ranking("b", tuple(["x"] * 7 + ["Genesis 1:1"]), 0.03),
    }
    r = score_rankings(questions, rankings)
    assert r["all"]["recall@5"] == 0.5
    assert r["all"]["recall@10"] == 1.0
    assert r["all"]["mrr@10"] == pytest.approx((1 / 3 + 1 / 8) / 2)
    assert r["all"]["all_gold@10"] == pytest.approx((0.5 + 1.0) / 2)
    assert r["lang:he"]["n"] == 1


def test_out_of_scope_and_goldless_questions_are_not_scored() -> None:
    questions = [q("a", ("Genesis 1:1",)), q("o", (), typ="out_of_scope"), q("n", (), typ="not_in_corpus")]
    r = score_rankings(questions, {"a": Ranking("a", ("Genesis 1:1",), 0.0)})
    assert r["all"]["n"] == 1


def test_verify_catches_missing_references_and_wrong_keywords() -> None:
    questions = [
        Question("a", "dev", "en", "source", "?", ("Genesis 1:1",), {"Genesis 1:1": ["בראשית"]}),
        Question("b", "dev", "en", "source", "?", ("Genesis 1:1",), {"Genesis 1:1": ["שמע"]}),
        Question("c", "test", "en", "source", "?", ("Exodus 1:1",), {}),
        Question("a", "dev", "en", "source", "?", ("Genesis 1:1",), {}),
    ]
    problems = verify_questions(questions, CHUNKS)
    assert problems == [
        "b: 'Genesis 1:1' does not contain 'שמע'",
        "c: unknown split test",
        "c: gold 'Exodus 1:1' is not in the corpus",
        "a: duplicate id",
    ]


def test_question_set_shape() -> None:
    """The committed set: at least 150 questions, both languages, a held-out split, out-of-scope
    prompts for the refusal rate, and sensitive questions."""
    questions = load_questions()
    assert len(questions) >= 150
    assert len({x.id for x in questions}) == len(questions)
    by_split = {s: [x for x in questions if x.split == s] for s in ("dev", "holdout")}
    assert len(by_split["holdout"]) > len(by_split["dev"]) > 50
    for split in by_split.values():
        assert {x.lang for x in split} == {"he", "en"}
        assert {"out_of_scope", "sensitive", "halacha", "source"} <= {x.type for x in split}
    assert all(x.gold for x in questions if x.type in ("source", "explain", "halacha", "sensitive"))
