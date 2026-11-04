from collections.abc import Sequence
from datetime import date

import pytest

from graphrag.cli.index import index_articles, run_indexing
from graphrag.domain.models import Article, IndexedChunk, IndexInfo


def article(n: int, abstract: str = "One sentence.", title: str = "Title") -> Article:
    return Article(f"2401.{n:05d}", f"{title} {n}", abstract, date(2025, 1, 1), ("cs.CL",), ())


class FakeEmbedder:
    dimensions = 2

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 1.0]


class FakeChunkStore:
    def __init__(self, articles: list[Article], fail_on_write: int | None = None) -> None:
        self.articles = articles
        self.chunks: dict[str, IndexedChunk] = {}
        self.schema_dimensions: list[int] = []
        self.writes: list[list[str]] = []
        self.fail_on_write = fail_on_write

    def ensure_chunk_schema(self, dimensions: int) -> None:
        self.schema_dimensions.append(dimensions)

    def ensure_index_info(self, info: IndexInfo) -> None:
        self.infos = [*getattr(self, "infos", []), info]

    def articles_without_chunks(self, limit: int | None = None) -> list[Article]:
        done = {c.chunk.arxiv_id for c in self.chunks.values()}
        todo = [a for a in self.articles if a.arxiv_id not in done]
        return todo if limit is None else todo[:limit]

    def upsert_chunks(self, chunks: Sequence[IndexedChunk]) -> None:
        if self.fail_on_write is not None and len(self.writes) == self.fail_on_write:
            raise RuntimeError("neo4j went away")
        self.writes.append(sorted({c.chunk.arxiv_id for c in chunks}))
        self.chunks.update({c.key: c for c in chunks})

    def chunk_count(self) -> int:
        return len(self.chunks)


def quiet(_message: str) -> None:
    pass


def long_abstract(sentences: int) -> str:
    return " ".join(
        f"Sentence number {i} has exactly forty words " + "x " * 32 + "end."
        for i in range(sentences)
    )


def test_index_articles_embeds_title_plus_chunk_and_keeps_chunk_text() -> None:
    embedder = FakeEmbedder()
    (indexed,) = index_articles([article(1, "Dense retrieval works.")], embedder)
    assert indexed.chunk.text == "Dense retrieval works."
    assert indexed.search_text == "Title 1. Dense retrieval works."
    assert embedder.calls == [["Title 1. Dense retrieval works."]]
    assert indexed.embedding == (float(len("Title 1. Dense retrieval works.")), 1.0)
    assert indexed.key == "2401.00001#0"


def test_every_chunk_of_a_long_abstract_gets_its_own_title() -> None:
    indexed = index_articles([article(1, long_abstract(8))], FakeEmbedder())
    assert len(indexed) > 1
    assert [c.chunk.position for c in indexed] == list(range(len(indexed)))
    assert all(c.search_text.startswith("Title 1. ") for c in indexed)


def test_chunks_are_matched_to_their_own_article_title() -> None:
    indexed = index_articles([article(1), article(2)], FakeEmbedder())
    assert [(c.chunk.arxiv_id, c.search_text) for c in indexed] == [
        ("2401.00001", "Title 1. One sentence."),
        ("2401.00002", "Title 2. One sentence."),
    ]


def test_run_creates_schema_with_embedder_dimensions_and_indexes_all() -> None:
    store = FakeChunkStore([article(n) for n in range(1, 6)])
    report = run_indexing(store, FakeEmbedder(), batch_size=2, log=quiet)
    assert store.schema_dimensions == [2]
    assert store.writes == [
        ["2401.00001", "2401.00002"],
        ["2401.00003", "2401.00004"],
        ["2401.00005"],
    ]
    assert (report.articles, report.chunks, report.total_chunks) == (5, 5, 5)


def test_rerun_indexes_nothing_more() -> None:
    store = FakeChunkStore([article(1), article(2)])
    run_indexing(store, FakeEmbedder(), log=quiet)
    embedder = FakeEmbedder()
    report = run_indexing(store, embedder, log=quiet)
    assert (report.articles, report.chunks, report.total_chunks) == (0, 0, 2)
    assert embedder.calls == []


def test_interrupted_run_resumes_with_the_remaining_articles() -> None:
    store = FakeChunkStore([article(n) for n in range(1, 6)], fail_on_write=1)
    with pytest.raises(RuntimeError, match="went away"):
        run_indexing(store, FakeEmbedder(), batch_size=2, log=quiet)
    store.fail_on_write = None
    report = run_indexing(store, FakeEmbedder(), batch_size=2, log=quiet)
    assert report.articles == 3
    assert report.total_chunks == 5


def test_limit_caps_the_number_of_articles() -> None:
    store = FakeChunkStore([article(n) for n in range(1, 6)])
    report = run_indexing(store, FakeEmbedder(), limit=3, log=quiet)
    assert report.articles == 3


def test_progress_is_logged_per_batch() -> None:
    lines: list[str] = []
    run_indexing(
        FakeChunkStore([article(1), article(2), article(3)]),
        FakeEmbedder(),
        batch_size=2,
        log=lines.append,
    )
    assert lines == [
        "chunks: 3 articles to index",
        "chunks: 2/3 articles indexed, 2 chunks",
        "chunks: 3/3 articles indexed, 3 chunks",
    ]


def test_indexing_records_the_fingerprint_when_given_and_before_writing() -> None:
    store = FakeChunkStore([])
    info = IndexInfo("m", "r", 2, "title. chunk")
    run_indexing(store, FakeEmbedder(), info=info, log=quiet)
    assert store.infos == [info]


def test_indexing_without_a_fingerprint_does_not_record_one() -> None:
    store = FakeChunkStore([])
    run_indexing(store, FakeEmbedder(), log=quiet)
    assert not hasattr(store, "infos")
