from collections.abc import Iterator, Sequence
from datetime import date

import pytest

from graphrag.cli.ingest import ARTICLE_BATCH, IngestionReport, run_ingestion
from graphrag.domain.models import Article, Author, GraphCounts, ReferenceRecord, WorkRecord


def article(n: int) -> Article:
    return Article(f"2501.{n:05d}", f"T{n}", "abs", date(2025, 1, 1), ("cs.CL",), ("Ann",))


def arxiv_id(n: int) -> str:
    return f"2501.{n:05d}"


class FakePapers:
    def __init__(self, count: int, fail_after: int | None = None) -> None:
        self.count = count
        self.fail_after = fail_after

    def iter_papers(self, start: date, end: date) -> Iterator[Article]:
        for n in range(1, self.count + 1):
            if self.fail_after is not None and n > self.fail_after:
                raise RuntimeError("arXiv went away")
            yield article(n)


def number(identifier: str) -> int:
    return int(identifier.split(".")[1])


class FakeAuthors:
    def __init__(self, known: set[int]) -> None:
        self.known = known
        self.calls: list[int] = []

    def fetch_works(self, arxiv_ids: Sequence[str]) -> list[WorkRecord]:
        self.calls.append(len(arxiv_ids))
        return [
            WorkRecord(i, f"W{number(i)}", (Author(f"A{number(i)}", "Au"),))
            for i in arxiv_ids
            if number(i) in self.known
        ]


class FakeCitations:
    def __init__(self, known: dict[int, tuple[int, ...]]) -> None:
        self.known = known
        self.calls: list[int] = []

    def fetch_references(self, arxiv_ids: Sequence[str]) -> list[ReferenceRecord]:
        self.calls.append(len(arxiv_ids))
        return [
            ReferenceRecord(i, tuple(arxiv_id(c) for c in self.known[number(i)]))
            for i in arxiv_ids
            if number(i) in self.known
        ]


class FakeStore:
    def __init__(self) -> None:
        self.articles: dict[str, Article] = {}
        self.events: list[str] = []
        self.cited: list[ReferenceRecord] = []

    def ensure_schema(self) -> None:
        self.events.append("schema")

    def upsert_articles(self, articles: Sequence[Article]) -> None:
        self.events.append(f"articles:{len(articles)}")
        for a in articles:
            self.articles.setdefault(a.arxiv_id, a)

    def upsert_works(self, records: Sequence[WorkRecord]) -> None:
        self.events.append(f"works:{len(records)}")

    def upsert_citations(self, records: Sequence[ReferenceRecord]) -> None:
        self.events.append(f"cites:{len(records)}")
        self.cited = list(records)

    def article_ids(self) -> list[str]:
        return sorted(self.articles)

    def counts(self) -> GraphCounts:
        return GraphCounts(len(self.articles), 0, 0, 0)


def run(
    papers: FakePapers,
    authors: FakeAuthors,
    citations: FakeCitations,
    store: FakeStore,
    limit: int | None = None,
) -> IngestionReport:
    return run_ingestion(
        papers,
        authors,
        citations,
        store,
        date(2025, 1, 1),
        date(2025, 12, 31),
        limit=limit,
        log=lambda _m: None,
    )


def test_full_run_reports_every_gap() -> None:
    store = FakeStore()
    authors = FakeAuthors({1, 2})  # article 3 unknown to OpenAlex
    citations = FakeCitations({1: (2,), 3: ()})  # article 2 unknown to Semantic Scholar
    report = run(FakePapers(3), authors, citations, store)
    assert report.articles_fetched == 3
    assert (report.authors_found, report.authors_missing) == (2, 1)
    assert (report.references_found, report.references_missing) == (2, 1)
    assert report.without_arxiv_references == 1
    assert store.cited == [
        ReferenceRecord(arxiv_id(1), (arxiv_id(2),)),
        ReferenceRecord(arxiv_id(3), ()),
    ]


def test_phases_are_ordered_articles_then_authors_then_citations() -> None:
    store = FakeStore()
    run(FakePapers(2), FakeAuthors({1, 2}), FakeCitations({1: (), 2: ()}), store)
    assert store.events == ["schema", "articles:2", "works:2", "cites:2"]


def test_limit_caps_the_test_lot() -> None:
    store = FakeStore()
    report = run(FakePapers(50), FakeAuthors(set()), FakeCitations({}), store, limit=7)
    assert report.articles_fetched == 7
    assert len(store.articles) == 7


def test_articles_are_written_page_by_page() -> None:
    store = FakeStore()
    run(FakePapers(ARTICLE_BATCH * 2 + 5), FakeAuthors(set()), FakeCitations({}), store)
    written = [e for e in store.events if e.startswith("articles:")]
    assert written == [f"articles:{ARTICLE_BATCH}", f"articles:{ARTICLE_BATCH}", "articles:5"]


def test_lookups_are_batched_by_100_for_both_sources() -> None:
    store = FakeStore()
    authors, citations = FakeAuthors(set()), FakeCitations({})
    run(FakePapers(250), authors, citations, store)
    assert authors.calls == [100, 100, 50]
    assert citations.calls == [100, 100, 50]


def test_interruption_keeps_written_pages_and_rerun_completes() -> None:
    store = FakeStore()
    broken = FakePapers(ARTICLE_BATCH + 20, fail_after=ARTICLE_BATCH + 10)
    with pytest.raises(RuntimeError, match="went away"):
        run(broken, FakeAuthors(set()), FakeCitations({}), store)
    assert len(store.articles) == ARTICLE_BATCH  # first full page survived
    run(FakePapers(ARTICLE_BATCH + 20), FakeAuthors(set()), FakeCitations({}), store)
    assert len(store.articles) == ARTICLE_BATCH + 20


def test_citation_source_failure_happens_after_articles_are_safe() -> None:
    class Broken:
        def fetch_references(self, arxiv_ids: Sequence[str]) -> list[ReferenceRecord]:
            raise RuntimeError("semantic scholar down")

    store = FakeStore()
    with pytest.raises(RuntimeError, match="semantic scholar down"):
        run_ingestion(
            FakePapers(3),
            FakeAuthors({1}),
            Broken(),
            store,
            date(2025, 1, 1),
            date(2025, 12, 31),
            log=lambda _m: None,
        )
    assert len(store.articles) == 3
    assert "cites:0" not in store.events  # nothing half-written on the citation side


def test_phase_two_uses_whole_stored_corpus_not_only_this_run() -> None:
    store = FakeStore()
    run(FakePapers(3), FakeAuthors(set()), FakeCitations({}), store)
    authors, citations = FakeAuthors(set()), FakeCitations({})
    run(FakePapers(0), authors, citations, store)  # nothing new fetched
    assert authors.calls == [3]
    assert citations.calls == [3]
