"""The `vector` and `hybrid` retrievers, and Reciprocal Rank Fusion.

Both searches return chunks; results are articles, each represented by its best chunk.
"""

from collections.abc import Sequence
from dataclasses import replace

from graphrag.domain.models import SearchHit
from graphrag.ports.embedding import Embedder
from graphrag.ports.retrieval import ChunkSearch

RRF_K = 60  # constant of the original RRF paper: dampens the gap between the top ranks
CANDIDATES = 50  # chunks fetched from each search before fusion (at least)
CHUNKS_PER_ARTICLE = 2  # an article has at most this many chunks


def fetch_depth(k: int, candidates: int = CANDIDATES) -> int:
    """Chunks to fetch for k articles: even if every article filled all its chunks, k remain."""
    return max(candidates, CHUNKS_PER_ARTICLE * k)


def best_chunk_per_article(hits: Sequence[SearchHit]) -> list[SearchHit]:
    """Keep the first (best ranked) chunk of each article, in order."""
    seen: set[str] = set()
    best = []
    for hit in hits:
        if hit.arxiv_id not in seen:
            seen.add(hit.arxiv_id)
            best.append(hit)
    return best


def rrf_fuse(rankings: Sequence[Sequence[SearchHit]], k: int = RRF_K) -> list[SearchHit]:
    """Fuse rankings of articles: each article scores the sum of 1 / (k + rank) (rank from 1).

    Each ranking is first reduced to one chunk per article, so an article with two chunks
    does not count twice. The chunk kept is the best ranked one across all rankings, the
    earlier ranking winning ties. Equal scores keep the order of first appearance.
    """
    scores: dict[str, float] = {}
    chunk: dict[str, tuple[int, SearchHit]] = {}  # arxiv_id -> (rank, hit)
    for ranking in rankings:
        for rank, hit in enumerate(best_chunk_per_article(ranking), start=1):
            scores[hit.arxiv_id] = scores.get(hit.arxiv_id, 0.0) + 1.0 / (k + rank)
            if hit.arxiv_id not in chunk or rank < chunk[hit.arxiv_id][0]:
                chunk[hit.arxiv_id] = (rank, hit)
    order = sorted(scores, key=lambda a: scores[a], reverse=True)  # stable on ties
    return [replace(chunk[a][1], score=scores[a]) for a in order]


class VectorRetriever:
    """Search by meaning only."""

    def __init__(
        self, search: ChunkSearch, embedder: Embedder, *, candidates: int = CANDIDATES
    ) -> None:
        self._search = search
        self._embedder = embedder
        self._candidates = candidates

    def retrieve(self, question: str, k: int) -> list[SearchHit]:
        vector = self._embedder.embed_query(question)
        hits = self._search.vector_search(vector, fetch_depth(k, self._candidates))
        return best_chunk_per_article(hits)[:k]


class KeywordRetriever:
    """Search by keywords only: a diagnostic, to see what each half brings to the fusion."""

    def __init__(self, search: ChunkSearch, *, candidates: int = CANDIDATES) -> None:
        self._search = search
        self._candidates = candidates

    def retrieve(self, question: str, k: int) -> list[SearchHit]:
        hits = self._search.keyword_search(question, fetch_depth(k, self._candidates))
        return best_chunk_per_article(hits)[:k]


class HybridRetriever:
    """Search by meaning and by keywords, fused with RRF."""

    def __init__(
        self,
        search: ChunkSearch,
        embedder: Embedder,
        *,
        candidates: int = CANDIDATES,
        rrf_k: int = RRF_K,
    ) -> None:
        self._search = search
        self._embedder = embedder
        self._candidates = candidates
        self._rrf_k = rrf_k

    def retrieve(self, question: str, k: int) -> list[SearchHit]:
        n = fetch_depth(k, self._candidates)
        by_meaning = self._search.vector_search(self._embedder.embed_query(question), n)
        by_keywords = self._search.keyword_search(question, n)
        return rrf_fuse([by_meaning, by_keywords], self._rrf_k)[:k]
