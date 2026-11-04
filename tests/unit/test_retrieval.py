from collections.abc import Sequence

import pytest

from graphrag.domain.models import SearchHit
from graphrag.pipeline.retrieval import (
    HybridRetriever,
    VectorRetriever,
    best_chunk_per_article,
    rrf_fuse,
)


def hit(arxiv_id: str, position: int = 0, score: float = 0.0) -> SearchHit:
    return SearchHit(arxiv_id, f"title {arxiv_id}", position, f"{arxiv_id}#{position}", score)


def ids(hits: Sequence[SearchHit]) -> list[str]:
    return [h.arxiv_id for h in hits]


def test_best_chunk_per_article_keeps_first_occurrence_in_order() -> None:
    hits = [hit("A", 1), hit("B"), hit("A", 0), hit("C")]
    assert [(h.arxiv_id, h.position) for h in best_chunk_per_article(hits)] == [
        ("A", 1),
        ("B", 0),
        ("C", 0),
    ]


def test_rrf_matches_the_worked_example() -> None:
    # meaning: A 1st, B 2nd, ..., C 8th; keywords: C 1st, D 2nd, ..., A 5th
    meaning = [hit("A"), hit("B"), hit("x1"), hit("x2"), hit("x3"), hit("x4"), hit("x5"), hit("C")]
    keywords = [hit("C"), hit("D"), hit("y1"), hit("y2"), hit("A")]
    fused = rrf_fuse([meaning, keywords])
    assert ids(fused)[:4] == ["A", "C", "B", "D"]
    scores = {h.arxiv_id: h.score for h in fused}
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 65)
    assert scores["C"] == pytest.approx(1 / 68 + 1 / 61)
    assert scores["B"] == pytest.approx(1 / 62)
    assert scores["D"] == pytest.approx(1 / 62)


def test_rrf_constant_is_configurable() -> None:
    (only,) = rrf_fuse([[hit("A")]], k=0)
    assert only.score == pytest.approx(1.0)


def test_an_article_with_two_chunks_counts_once_per_ranking() -> None:
    fused = rrf_fuse([[hit("A", 0), hit("A", 1), hit("B")]])
    scores = {h.arxiv_id: h.score for h in fused}
    assert scores["A"] == pytest.approx(1 / 61)
    assert scores["B"] == pytest.approx(1 / 62)  # B is 2nd once A's duplicate is dropped


def test_the_kept_chunk_is_the_best_ranked_one() -> None:
    meaning = [hit("Z"), hit("Z2"), hit("A", 0)]  # A's chunk 0 is 3rd here
    keywords = [hit("A", 1)]  # A's chunk 1 is 1st here
    (a,) = [h for h in rrf_fuse([meaning, keywords]) if h.arxiv_id == "A"]
    assert a.position == 1


def test_on_equal_rank_the_first_ranking_chunk_wins() -> None:
    (a,) = rrf_fuse([[hit("A", 0)], [hit("A", 1)]])
    assert a.position == 0


def test_ties_keep_first_appearance_order() -> None:
    assert ids(rrf_fuse([[hit("A")], [hit("B")]])) == ["A", "B"]


def test_empty_rankings() -> None:
    assert rrf_fuse([[], []]) == []


class FakeEmbedder:
    dimensions = 2

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError("retrieval must not embed documents")

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]


class FakeSearch:
    def __init__(self, by_meaning: list[SearchHit], by_keywords: list[SearchHit]) -> None:
        self.by_meaning = by_meaning
        self.by_keywords = by_keywords
        self.calls: list[tuple[str, object, int]] = []

    def vector_search(self, vector: Sequence[float], k: int) -> list[SearchHit]:
        self.calls.append(("vector", list(vector), k))
        return self.by_meaning[:k]

    def keyword_search(self, text: str, k: int) -> list[SearchHit]:
        self.calls.append(("keyword", text, k))
        return self.by_keywords[:k]


def test_vector_retriever_embeds_the_question_and_dedupes() -> None:
    search = FakeSearch([hit("A", 0, 0.9), hit("A", 1, 0.8), hit("B", 0, 0.7)], [])
    embedder = FakeEmbedder()
    result = VectorRetriever(search, embedder, candidates=20).retrieve("q?", k=5)
    assert embedder.queries == ["q?"]
    assert search.calls == [("vector", [1.0, 0.0], 20)]
    assert [(h.arxiv_id, h.score) for h in result] == [("A", 0.9), ("B", 0.7)]


def test_retrievers_fetch_at_least_k_candidates_and_cut_to_k() -> None:
    many = [hit(f"P{i}") for i in range(30)]
    search = FakeSearch(many, many)
    assert len(VectorRetriever(search, FakeEmbedder(), candidates=5).retrieve("q", 25)) == 25
    assert search.calls[-1][2] == 25
    assert len(HybridRetriever(search, FakeEmbedder(), candidates=20).retrieve("q", 3)) == 3
    assert [c[2] for c in search.calls[-2:]] == [20, 20]


def test_hybrid_retriever_sends_the_raw_question_to_both_searches_and_fuses() -> None:
    search = FakeSearch([hit("A"), hit("B")], [hit("C"), hit("A")])
    embedder = FakeEmbedder()
    result = HybridRetriever(search, embedder).retrieve("graph RAG?", k=10)
    assert embedder.queries == ["graph RAG?"]
    assert [c[0:2] for c in search.calls] == [("vector", [1.0, 0.0]), ("keyword", "graph RAG?")]
    assert ids(result) == ["A", "C", "B"]  # A is in both lists
    assert result[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_hybrid_uses_its_rrf_constant() -> None:
    search = FakeSearch([hit("A")], [])
    (a,) = HybridRetriever(search, FakeEmbedder(), rrf_k=0).retrieve("q", 1)
    assert a.score == pytest.approx(1.0)
