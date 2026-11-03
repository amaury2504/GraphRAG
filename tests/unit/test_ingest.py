from collections.abc import Iterator, Sequence
from datetime import date
from pathlib import Path

import pytest

from graphrag.adapters.http import RequestFailedError
from graphrag.cli.ingest import (
    ARTICLE_BATCH,
    EXIT_INCOMPLETE,
    CitationReport,
    IngestionReport,
    _parse_args,
    exit_code,
    ledger_from_args,
    make_ledger,
    run_citations,
    run_ingestion,
)
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
        self.cited.extend(records)

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
    assert not any(e.startswith("cites:") for e in store.events)


def test_phase_two_uses_whole_stored_corpus_not_only_this_run() -> None:
    store = FakeStore()
    run(FakePapers(3), FakeAuthors(set()), FakeCitations({}), store)
    authors, citations = FakeAuthors(set()), FakeCitations({})
    run(FakePapers(0), authors, citations, store)  # nothing new fetched
    assert authors.calls == [3]
    assert citations.calls == [3]


def test_citations_are_written_batch_by_batch() -> None:
    store = FakeStore()
    known: dict[int, tuple[int, ...]] = dict.fromkeys(range(1, 251), ())
    run(FakePapers(250), FakeAuthors(set()), FakeCitations(known), store)
    assert [e for e in store.events if e.startswith("cites:")] == [
        "cites:100",
        "cites:100",
        "cites:50",
    ]


def test_interruption_during_citations_keeps_the_batches_already_written() -> None:
    class FailsOnSecondBatch:
        def __init__(self) -> None:
            self.calls = 0

        def fetch_references(self, arxiv_ids: Sequence[str]) -> list[ReferenceRecord]:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("semantic scholar down")
            return [ReferenceRecord(i, ()) for i in arxiv_ids]

    store = FakeStore()
    with pytest.raises(RuntimeError, match="semantic scholar down"):
        run_ingestion(
            FakePapers(250),
            FakeAuthors(set()),
            FailsOnSecondBatch(),
            store,
            date(2025, 1, 1),
            date(2025, 12, 31),
            log=lambda _m: None,
        )
    assert len(store.cited) == 100


def test_progress_is_logged_for_each_citation_batch() -> None:
    messages: list[str] = []
    run_ingestion(
        FakePapers(150),
        FakeAuthors(set()),
        FakeCitations(dict.fromkeys(range(1, 151), ())),
        FakeStore(),
        date(2025, 1, 1),
        date(2025, 12, 31),
        log=messages.append,
    )
    refs = [m for m in messages if m.startswith("references")]
    assert refs == [
        "references (semantic scholar): 0 already checked, 150 to look up",
        "references (semantic scholar): 100/150 looked up, 100 articles known so far",
        "references (semantic scholar): 150/150 looked up, 150 articles known so far",
    ]


def test_default_log_flushes_immediately(capsys: pytest.CaptureFixture[str]) -> None:
    from unittest.mock import patch

    with patch("builtins.print") as fake_print:
        from graphrag.cli.ingest import log_line

        log_line("hello")
    fake_print.assert_called_once_with("hello", flush=True)


class FailingBatches:
    """Raises RequestFailedError on the given (1-based) call numbers."""

    def __init__(self, failing_calls: set[int]) -> None:
        self.failing_calls = failing_calls
        self.calls = 0

    def fetch_references(self, arxiv_ids: Sequence[str]) -> list[ReferenceRecord]:
        self.calls += 1
        if self.calls in self.failing_calls:
            raise RequestFailedError("HTTP 429")
        return [ReferenceRecord(i, ()) for i in arxiv_ids]


def test_exhausted_batch_is_skipped_and_the_run_goes_on() -> None:
    store = FakeStore()
    store.upsert_articles([article(n) for n in range(1, 251)])
    source = FailingBatches({2})
    report = run_citations(source, store, store.article_ids(), log=lambda _m: None)
    assert source.calls == 3
    assert len(store.cited) == 150
    assert report == CitationReport(
        found=150, missing=0, without_arxiv_references=150, failed_batches=1
    )


def test_skipped_ids_are_not_counted_as_missing() -> None:
    store = FakeStore()
    store.upsert_articles([article(n) for n in range(1, 4)])
    report = run_citations(FailingBatches({1}), store, store.article_ids(), log=lambda _m: None)
    assert (report.found, report.missing, report.failed_batches) == (0, 0, 1)


def test_skipped_batch_is_logged() -> None:
    store = FakeStore()
    store.upsert_articles([article(n) for n in range(1, 4)])
    messages: list[str] = []
    run_citations(FailingBatches({1}), store, store.article_ids(), log=messages.append)
    assert messages[1:] == ["references (semantic scholar): 3/3 batch skipped (HTTP 429)"]


def test_full_run_reports_failed_citation_batches() -> None:
    class AlwaysBusy:
        def fetch_references(self, arxiv_ids: Sequence[str]) -> list[ReferenceRecord]:
            raise RequestFailedError("HTTP 429")

    store = FakeStore()
    report = run_ingestion(
        FakePapers(3),
        FakeAuthors(set()),
        AlwaysBusy(),
        store,
        date(2025, 1, 1),
        date(2025, 12, 31),
        log=lambda _m: None,
    )
    assert report.citation_batches_failed == 1
    assert len(store.articles) == 3


@pytest.mark.parametrize(("failed", "code"), [(0, 0), (1, EXIT_INCOMPLETE), (5, EXIT_INCOMPLETE)])
def test_exit_code_tells_whether_a_rerun_is_needed(failed: int, code: int) -> None:
    assert exit_code(failed) == code
    assert EXIT_INCOMPLETE not in (0, 1, 2)  # distinct from success, crash and argparse errors


def test_citations_only_flag() -> None:
    assert _parse_args(["--citations-only"]).citations_only is True
    assert _parse_args([]).citations_only is False


class FakeLedger:
    def __init__(self, checked: set[str] | None = None) -> None:
        self.done: set[str] = set(checked or ())
        self.marks: list[list[str]] = []

    def checked(self) -> set[str]:
        return set(self.done)

    def mark(self, arxiv_ids: Sequence[str]) -> None:
        self.marks.append(list(arxiv_ids))
        self.done.update(arxiv_ids)


def stored(count: int) -> FakeStore:
    store = FakeStore()
    store.upsert_articles([article(n) for n in range(1, count + 1)])
    return store


def test_already_checked_ids_are_not_looked_up_again() -> None:
    store = stored(250)
    ledger = FakeLedger({arxiv_id(n) for n in range(1, 201)})
    source = FakeCitations(dict.fromkeys(range(1, 251), ()))
    run_citations(source, store, store.article_ids(), ledger=ledger, log=lambda _m: None)
    assert source.calls == [50]


def test_every_id_of_a_written_batch_is_marked_even_unknown_ones() -> None:
    store = stored(3)
    ledger = FakeLedger()
    source = FakeCitations({1: ()})  # 2 and 3 unknown to the source
    run_citations(source, store, store.article_ids(), ledger=ledger, log=lambda _m: None)
    assert ledger.done == {arxiv_id(1), arxiv_id(2), arxiv_id(3)}


def test_skipped_batch_is_not_marked() -> None:
    store = stored(250)
    ledger = FakeLedger()
    run_citations(
        FailingBatches({2}), store, store.article_ids(), ledger=ledger, log=lambda _m: None
    )
    assert ledger.done == {arxiv_id(n) for n in [*range(1, 101), *range(201, 251)]}


def test_rerun_after_a_skipped_batch_only_redoes_that_batch() -> None:
    store = stored(250)
    ledger = FakeLedger()
    run_citations(
        FailingBatches({2}), store, store.article_ids(), ledger=ledger, log=lambda _m: None
    )
    source = FakeCitations(dict.fromkeys(range(1, 251), ()))
    report = run_citations(source, store, store.article_ids(), ledger=ledger, log=lambda _m: None)
    assert source.calls == [100]
    assert report.failed_batches == 0


def test_mark_happens_after_the_write() -> None:
    order: list[str] = []

    class Store(FakeStore):
        def upsert_citations(self, records: Sequence[ReferenceRecord]) -> None:
            order.append("write")

    class Ledger(FakeLedger):
        def mark(self, arxiv_ids: Sequence[str]) -> None:
            order.append("mark")

    store = Store()
    store.upsert_articles([article(1)])
    run_citations(
        FakeCitations({1: ()}), store, store.article_ids(), ledger=Ledger(), log=lambda _m: None
    )
    assert order == ["write", "mark"]


def test_crash_while_writing_leaves_the_batch_unmarked() -> None:
    class Store(FakeStore):
        def upsert_citations(self, records: Sequence[ReferenceRecord]) -> None:
            raise RuntimeError("neo4j down")

    store = Store()
    store.upsert_articles([article(1)])
    ledger = FakeLedger()
    with pytest.raises(RuntimeError, match="neo4j down"):
        run_citations(
            FakeCitations({1: ()}), store, store.article_ids(), ledger=ledger, log=lambda _m: None
        )
    assert ledger.done == set()


def test_start_message_says_what_is_left() -> None:
    store = stored(3)
    messages: list[str] = []
    run_citations(
        FakeCitations({}),
        store,
        store.article_ids(),
        ledger=FakeLedger({arxiv_id(1)}),
        log=messages.append,
    )
    assert messages[0] == "references (semantic scholar): 1 already checked, 2 to look up"


def test_full_ingestion_passes_the_ledger_through() -> None:
    store = FakeStore()
    ledger = FakeLedger({arxiv_id(1)})
    citations = FakeCitations({})
    run_ingestion(
        FakePapers(3),
        FakeAuthors(set()),
        citations,
        store,
        date(2025, 1, 1),
        date(2025, 12, 31),
        ledger=ledger,
        log=lambda _m: None,
    )
    assert citations.calls == [2]


def test_recheck_ledger_forgets_but_still_records() -> None:
    inner = FakeLedger({arxiv_id(1)})
    ledger = make_ledger(True, inner)
    assert ledger.checked() == set()
    ledger.mark([arxiv_id(2)])
    assert inner.done == {arxiv_id(1), arxiv_id(2)}
    assert make_ledger(False, inner) is inner


def test_recheck_flag() -> None:
    assert _parse_args(["--recheck-citations"]).recheck_citations is True
    assert _parse_args([]).recheck_citations is False


@pytest.mark.parametrize(("argv", "remembers"), [([], True), (["--recheck-citations"], False)])
def test_ledger_from_args_honours_recheck(tmp_path: Path, argv: list[str], remembers: bool) -> None:
    path = tmp_path / "checked.txt"
    path.write_text(arxiv_id(1) + "\n", encoding="utf-8")
    ledger = ledger_from_args(_parse_args(argv), path)
    assert ledger.checked() == ({arxiv_id(1)} if remembers else set())
    ledger.mark([arxiv_id(2)])
    assert arxiv_id(2) in path.read_text(encoding="utf-8")
