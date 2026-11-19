from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime

import pytest

from graphrag.cli.index import _parse_args, index_articles, run_indexing, run_rollback
from graphrag.domain.hashing import article_hash
from graphrag.domain.models import Article, ArticleIndexState, IndexedChunk, IndexInfo
from graphrag.domain.versioning import VersionInfo


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
    """Chunks are immutable and never deleted; a version is a manifest of the latest chunk keys."""

    def __init__(self, articles: list[Article], fail_on_write: int | None = None) -> None:
        self.articles = articles
        self.chunks: dict[str, IndexedChunk] = {}
        self.hashes: dict[str, str] = {}
        self.article_keys: dict[str, set[str]] = {}
        self.manifests: list[frozenset[str]] = []
        self.current: int | None = None  # index into `manifests`
        self.schema_dimensions: list[int] = []
        self.writes: list[list[str]] = []
        self.fail_on_write = fail_on_write

    def ensure_chunk_schema(self, dimensions: int) -> None:
        self.schema_dimensions.append(dimensions)

    def ensure_index_info(self, info: IndexInfo) -> None:
        self.infos = [*getattr(self, "infos", []), info]

    def articles_with_index_state(self) -> list[ArticleIndexState]:
        return [ArticleIndexState(a, self.hashes.get(a.arxiv_id)) for a in self.articles]

    def upsert_chunks(
        self, chunks: Sequence[IndexedChunk], article_hashes: Mapping[str, str] | None = None
    ) -> None:
        if self.fail_on_write is not None and len(self.writes) == self.fail_on_write:
            raise RuntimeError("neo4j went away")
        owners = article_hashes or {}
        self.writes.append(sorted(owners or {c.chunk.arxiv_id for c in chunks}))
        self.chunks.update({c.key: c for c in chunks if c.key not in self.chunks})  # immutable
        for article_id in owners:
            self.article_keys[article_id] = {
                c.key for c in chunks if c.chunk.arxiv_id == article_id
            }
        self.hashes.update(owners)

    def publish_version(self) -> VersionInfo:
        self.manifests.append(frozenset().union(*self.article_keys.values()))
        self.current = len(self.manifests) - 1
        return self._info(self.current)

    def current_version(self) -> VersionInfo | None:
        return None if self.current is None else self._info(self.current)

    def rollback(self) -> VersionInfo:
        raise NotImplementedError  # the indexing pipeline never rolls back

    def _info(self, index: int) -> VersionInfo:
        created = datetime(2026, 11, 18, tzinfo=UTC)
        return VersionInfo(index + 1, created, len(self.manifests[index]))

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
    assert indexed.key.startswith("2401.00001#0#")


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
    assert report.version == 1


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
        "chunks: version 1 published (3 chunks)",
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


def test_an_article_whose_text_changed_is_indexed_again() -> None:
    store = FakeChunkStore([article(1), article(2)])
    run_indexing(store, FakeEmbedder(), log=quiet)
    store.articles[0] = article(1, abstract="A new abstract.")
    embedder = FakeEmbedder()
    report = run_indexing(store, embedder, log=quiet)
    assert report.articles == 1
    assert embedder.calls == [["Title 1. A new abstract."]]
    texts = {c.chunk.text for c in store.chunks.values()}
    assert texts == {"One sentence.", "A new abstract."}  # the old chunk is kept, not rewritten


def test_a_shorter_article_drops_its_extra_chunks_from_the_new_version_only() -> None:
    store = FakeChunkStore([article(1, long_abstract(8))])
    first = run_indexing(store, FakeEmbedder(), log=quiet)
    assert first.chunks > 1
    store.articles[0] = article(1, "Short now.")
    report = run_indexing(store, FakeEmbedder(), log=quiet)
    assert report.total_chunks == first.chunks + 1  # nothing is deleted: that is the GC's job
    assert len(store.manifests[0]) == first.chunks
    assert len(store.manifests[1]) == 1
    assert store.manifests[0].isdisjoint(store.manifests[1])


def test_an_article_left_with_no_chunk_is_absent_from_the_new_version() -> None:
    store = FakeChunkStore([article(1), article(2)])
    run_indexing(store, FakeEmbedder(), log=quiet)
    store.articles[0] = article(1, abstract="   ")
    report = run_indexing(store, FakeEmbedder(), log=quiet)
    assert (report.chunks, report.total_chunks) == (0, 2)
    assert len(store.manifests[1]) == 1
    assert store.hashes["2401.00001"] == article_hash(store.articles[0])


def test_the_hash_is_recorded_with_the_chunks_and_the_rerun_is_idempotent() -> None:
    store = FakeChunkStore([article(1), article(2)])
    run_indexing(store, FakeEmbedder(), log=quiet)
    assert store.hashes == {a.arxiv_id: article_hash(a) for a in store.articles}
    snapshot = dict(store.chunks)
    writes = len(store.writes)
    report = run_indexing(store, FakeEmbedder(), log=quiet)
    assert (report.articles, report.version) == (0, None)
    assert len(store.manifests) == 1  # no change, no new version
    assert store.chunks == snapshot
    assert len(store.writes) == writes


def test_an_article_indexed_before_hashes_existed_is_indexed_once_more() -> None:
    store = FakeChunkStore([article(1)])
    run_indexing(store, FakeEmbedder(), log=quiet)
    del store.hashes["2401.00001"]  # a database built before this feature
    assert run_indexing(store, FakeEmbedder(), log=quiet).articles == 1
    assert run_indexing(store, FakeEmbedder(), log=quiet).articles == 0


def test_a_new_chunker_version_reindexes_every_article(monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeChunkStore([article(1), article(2)])
    run_indexing(store, FakeEmbedder(), log=quiet)
    monkeypatch.setattr("graphrag.domain.hashing.CHUNKER_VERSION", "next")
    assert run_indexing(store, FakeEmbedder(), log=quiet).articles == 2


def test_limit_applies_to_the_stale_articles_only() -> None:
    store = FakeChunkStore([article(n) for n in range(1, 5)])
    run_indexing(store, FakeEmbedder(), limit=2, log=quiet)  # articles 1 and 2 are now current
    store.articles[0] = article(1, "Changed.")  # stale again: stale are now 1, 3 and 4
    report = run_indexing(store, FakeEmbedder(), limit=2, log=quiet)
    assert report.articles == 2
    assert sorted(store.writes[-1]) == ["2401.00001", "2401.00003"]


def test_an_interruption_after_a_change_leaves_the_old_chunks_and_resumes() -> None:
    store = FakeChunkStore([article(1, long_abstract(8)), article(2)])
    run_indexing(store, FakeEmbedder(), log=quiet)
    old_chunks = dict(store.chunks)
    store.articles[0] = article(1, "Short now.")
    store.fail_on_write = len(store.writes)
    with pytest.raises(RuntimeError, match="went away"):
        run_indexing(store, FakeEmbedder(), log=quiet)
    assert store.chunks == old_chunks  # nothing half written
    store.fail_on_write = None
    assert run_indexing(store, FakeEmbedder(), log=quiet).articles == 1


def test_a_run_publishes_one_version_after_all_its_batches() -> None:
    store = FakeChunkStore([article(n) for n in range(1, 6)])
    report = run_indexing(store, FakeEmbedder(), batch_size=2, log=quiet)
    assert len(store.writes) == 3
    assert len(store.manifests) == 1  # one snapshot, not one per batch
    assert len(store.manifests[0]) == 5
    assert report.version == 1


def test_a_changed_article_gives_a_new_version_that_shares_the_unchanged_chunks() -> None:
    store = FakeChunkStore([article(1), article(2)])
    run_indexing(store, FakeEmbedder(), log=quiet)
    store.articles[0] = article(1, "A new abstract.")
    report = run_indexing(store, FakeEmbedder(), log=quiet)
    first, second = store.manifests
    assert report.version == 2
    assert len(first & second) == 1  # article 2's chunk, shared and not copied
    assert len(first | second) == 3
    assert store.current_version() == VersionInfo(2, datetime(2026, 11, 18, tzinfo=UTC), 2)


def test_an_interrupted_run_publishes_nothing_and_the_resumed_run_publishes_everything() -> None:
    store = FakeChunkStore([article(n) for n in range(1, 6)], fail_on_write=1)
    with pytest.raises(RuntimeError, match="went away"):
        run_indexing(store, FakeEmbedder(), batch_size=2, log=quiet)
    assert store.current_version() is None  # the first batch is written but not readable
    store.fail_on_write = None
    report = run_indexing(store, FakeEmbedder(), batch_size=2, log=quiet)
    assert report.version == 1
    assert len(store.manifests[0]) == 5


def test_a_database_without_any_version_gets_its_first_one_even_with_nothing_to_index() -> None:
    store = FakeChunkStore([article(1)])
    run_indexing(store, FakeEmbedder(), log=quiet)
    store.manifests.clear()
    store.current = None  # chunks indexed before versions existed
    report = run_indexing(store, FakeEmbedder(), log=quiet)
    assert (report.articles, report.version) == (0, 1)
    assert len(store.manifests[0]) == 1


def test_the_log_announces_the_published_version() -> None:
    lines: list[str] = []
    run_indexing(FakeChunkStore([article(1)]), FakeEmbedder(), log=lines.append)
    assert lines[-1] == "chunks: version 1 published (1 chunks)"


def test_the_rollback_option_is_off_by_default() -> None:
    assert _parse_args([]).rollback is False
    assert _parse_args(["--rollback"]).rollback is True


def test_a_rollback_is_delegated_to_the_store_and_logged() -> None:
    class RollingBack(FakeChunkStore):
        def rollback(self) -> VersionInfo:
            return VersionInfo(1, datetime(2026, 11, 18, tzinfo=UTC), 7)

    lines: list[str] = []
    version = run_rollback(RollingBack([]), log=lines.append)
    assert version.id == 1
    assert lines == ["chunks: rolled back to version 1 (7 chunks)"]


def test_nothing_is_logged_about_a_version_when_none_is_published() -> None:
    store = FakeChunkStore([article(1)])
    run_indexing(store, FakeEmbedder(), log=quiet)
    lines: list[str] = []
    run_indexing(store, FakeEmbedder(), log=lines.append)
    assert lines == ["chunks: 0 articles to index"]
