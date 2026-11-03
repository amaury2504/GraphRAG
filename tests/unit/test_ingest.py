from collections.abc import Iterator, Sequence
from datetime import date

import pytest

from graphrag.cli.ingest import ARTICLE_BATCH, IngestionReport, run_ingestion
from graphrag.domain.models import Article, Author, GraphCounts, WorkRecord


def article(n: int) -> Article:
    return Article(f"2501.{n:05d}", f"T{n}", "abs", date(2025, 1, 1), ("cs.CL",), ("Ann",))


def work(n: int, refs: tuple[str, ...] = ()) -> WorkRecord:
    return WorkRecord(f"2501.{n:05d}", f"W{n}", (Author(f"A{n}", f"Au{n}"),), refs)


class FakePapers:
    def __init__(self, count: int, fail_after: int | None = None) -> None:
        self.count = count
        self.fail_after = fail_after

    def iter_papers(self, start: date, end: date) -> Iterator[Article]:
        for n in range(1, self.count + 1):
            if self.fail_after is not None and n > self.fail_after:
                raise RuntimeError("arXiv went away")
            yield article(n)


class FakeCitations:
    def __init__(self, known: dict[int, tuple[str, ...]]) -> None:
        self.known = known
        self.calls: list[int] = []

    def fetch_works(self, arxiv_ids: Sequence[str]) -> list[WorkRecord]:
        self.calls.append(len(arxiv_ids))
        out = []
        for arxiv_id in arxiv_ids:
            n = int(arxiv_id.split(".")[1])
            if n in self.known:
                out.append(work(n, self.known[n]))
        return out


class FakeStore:
    def __init__(self) -> None:
        self.articles: dict[str, Article] = {}
        self.works: dict[str, WorkRecord] = {}
        self.cite_rows: list[WorkRecord] = []
        self.events: list[str] = []

    def ensure_schema(self) -> None:
        self.events.append("schema")

    def upsert_articles(self, articles: Sequence[Article]) -> None:
        self.events.append(f"articles:{len(articles)}")
        for a in articles:
            self.articles.setdefault(a.arxiv_id, a)

    def upsert_works(self, records: Sequence[WorkRecord]) -> None:
        self.events.append(f"works:{len(records)}")
        self.works.update({r.arxiv_id: r for r in records})

    def upsert_citations(self, records: Sequence[WorkRecord]) -> None:
        self.events.append(f"cites:{len(records)}")
        self.cite_rows = list(records)

    def article_ids(self) -> list[str]:
        return sorted(self.articles)

    def counts(self) -> GraphCounts:
        return GraphCounts(len(self.articles), 0, 0, 0)


def run(
    papers: FakePapers,
    citations: FakeCitations,
    store: FakeStore,
    limit: int | None = None,
) -> IngestionReport:
    return run_ingestion(
        papers,
        citations,
        store,
        date(2025, 1, 1),
        date(2025, 12, 31),
        limit=limit,
        log=lambda _m: None,
    )


def test_full_run_reports_missing_and_reference_less_works() -> None:
    store = FakeStore()
    citations = FakeCitations({1: ("W2",), 2: ()})  # article 3 unknown to OpenAlex
    report = run(FakePapers(3), citations, store)
    assert report.articles_fetched == 3
    assert report.works_found == 2
    assert report.works_missing == 1
    assert report.works_without_references == 1
    assert store.events[0] == "schema"
    assert store.events[-1] == "cites:2"


def test_phases_are_ordered_articles_then_works_then_citations() -> None:
    store = FakeStore()
    run(FakePapers(2), FakeCitations({1: (), 2: ()}), store)
    assert store.events == ["schema", "articles:2", "works:2", "cites:2"]


def test_limit_caps_the_test_lot() -> None:
    store = FakeStore()
    report = run(FakePapers(50), FakeCitations({}), store, limit=7)
    assert report.articles_fetched == 7
    assert len(store.articles) == 7


def test_articles_are_written_page_by_page() -> None:
    store = FakeStore()
    run(FakePapers(ARTICLE_BATCH * 2 + 5), FakeCitations({}), store)
    written = [e for e in store.events if e.startswith("articles:")]
    assert written == [f"articles:{ARTICLE_BATCH}", f"articles:{ARTICLE_BATCH}", "articles:5"]


def test_openalex_lookups_are_batched_by_100() -> None:
    store = FakeStore()
    citations = FakeCitations({})
    run(FakePapers(250), citations, store)
    assert citations.calls == [100, 100, 50]


def test_interruption_keeps_written_pages_and_rerun_completes() -> None:
    store = FakeStore()
    with pytest.raises(RuntimeError, match="went away"):
        run(FakePapers(ARTICLE_BATCH + 20, fail_after=ARTICLE_BATCH + 10), FakeCitations({}), store)
    assert len(store.articles) == ARTICLE_BATCH  # first full page survived
    run(FakePapers(ARTICLE_BATCH + 20), FakeCitations({}), store)
    assert len(store.articles) == ARTICLE_BATCH + 20


def test_phase_two_uses_whole_stored_corpus_not_only_this_run() -> None:
    store = FakeStore()
    run(FakePapers(3), FakeCitations({}), store)
    citations = FakeCitations({})
    run(FakePapers(0), citations, store)  # nothing new fetched
    assert citations.calls == [3]
