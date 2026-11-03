import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import islice

import httpx

from graphrag.adapters.arxiv import ArxivSource
from graphrag.adapters.http import RequestFailedError
from graphrag.adapters.neo4j_store import Neo4jStore
from graphrag.adapters.openalex import BATCH_SIZE, OpenAlexSource
from graphrag.adapters.semantic_scholar import BATCH_SIZE as SEMANTIC_SCHOLAR_BATCH
from graphrag.adapters.semantic_scholar import SemanticScholarSource
from graphrag.domain.models import Article, GraphCounts, ReferenceRecord, WorkRecord
from graphrag.ports.ingestion import AuthorSource, CitationSource, GraphStore, PaperSource
from graphrag.settings import Settings

ARTICLE_BATCH = 100
EXIT_INCOMPLETE = 3  # some citation batches were skipped; rerun later


def log_line(message: str) -> None:
    """Print at once: output redirected to a file is otherwise buffered until the end."""
    print(message, flush=True)


@dataclass(frozen=True, slots=True)
class IngestionReport:
    articles_fetched: int
    authors_found: int
    authors_missing: int
    references_found: int
    references_missing: int
    without_arxiv_references: int
    citation_batches_failed: int
    counts: GraphCounts


@dataclass(frozen=True, slots=True)
class CitationReport:
    found: int
    missing: int  # looked up but unknown to the source
    without_arxiv_references: int
    failed_batches: int  # retries exhausted; these ids were not looked up


def run_ingestion(
    papers: PaperSource,
    authors: AuthorSource,
    citations: CitationSource,
    store: GraphStore,
    start: date,
    end: date,
    *,
    limit: int | None = None,
    log: Callable[[str], None] = log_line,
) -> IngestionReport:
    store.ensure_schema()

    # Phase 1: articles. Written page by page, so an interruption loses at most one page.
    fetched = 0
    stream = papers.iter_papers(start, end)
    if limit is not None:
        stream = islice(stream, limit)
    batch: list[Article] = []
    for article in stream:
        batch.append(article)
        fetched += 1
        if len(batch) == ARTICLE_BATCH:
            store.upsert_articles(batch)
            batch = []
            log(f"articles: {fetched} fetched")
    if batch:
        store.upsert_articles(batch)
    log(f"articles: {fetched} fetched (done)")

    # Phase 2: authors and citations for the whole stored corpus, so reruns stay consistent.
    ids = store.article_ids()
    works: list[WorkRecord] = []
    for offset in range(0, len(ids), BATCH_SIZE):
        found = authors.fetch_works(ids[offset : offset + BATCH_SIZE])
        store.upsert_works(found)
        works.extend(found)
        log(f"authors (openalex): {min(offset + BATCH_SIZE, len(ids))}/{len(ids)} looked up")

    cites = run_citations(citations, store, ids, log=log)

    return IngestionReport(
        articles_fetched=fetched,
        authors_found=len(works),
        authors_missing=len(ids) - len(works),
        references_found=cites.found,
        references_missing=cites.missing,
        without_arxiv_references=cites.without_arxiv_references,
        citation_batches_failed=cites.failed_batches,
        counts=store.counts(),
    )


def run_citations(
    citations: CitationSource,
    store: GraphStore,
    ids: Sequence[str],
    *,
    log: Callable[[str], None] = log_line,
) -> CitationReport:
    """Look up references of `ids` and write them batch by batch.

    Every article is stored before this runs, so citation targets exist. Writing per batch
    keeps progress visible in the database and survives an interruption. A batch whose
    retries are exhausted is skipped and counted, so one saturated window does not stop the
    run; any other error still stops it.
    """
    references: list[ReferenceRecord] = []
    failed_batches = 0
    failed_ids = 0
    for offset in range(0, len(ids), SEMANTIC_SCHOLAR_BATCH):
        batch = ids[offset : offset + SEMANTIC_SCHOLAR_BATCH]
        done = min(offset + SEMANTIC_SCHOLAR_BATCH, len(ids))
        try:
            found_refs = citations.fetch_references(batch)
        except RequestFailedError as exc:
            failed_batches += 1
            failed_ids += len(batch)
            log(f"references (semantic scholar): {done}/{len(ids)} batch skipped ({exc})")
            continue
        store.upsert_citations(found_refs)
        references.extend(found_refs)
        log(
            f"references (semantic scholar): {done}/{len(ids)} looked up, "
            f"{len(references)} articles known so far"
        )
    return CitationReport(
        found=len(references),
        missing=len(ids) - failed_ids - len(references),
        without_arxiv_references=sum(1 for r in references if not r.cited_arxiv_ids),
        failed_batches=failed_batches,
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    today = date.today()
    parser = argparse.ArgumentParser(description="Ingest arXiv RAG papers into Neo4j.")
    parser.add_argument("--since", type=date.fromisoformat, default=today - timedelta(days=365))
    parser.add_argument("--until", type=date.fromisoformat, default=today)
    parser.add_argument("--limit", type=int, default=None, help="stop after N articles (test lot)")
    parser.add_argument(
        "--citations-only",
        action="store_true",
        help="skip arXiv and OpenAlex; look up citations of the articles already stored",
    )
    return parser.parse_args(argv)


def exit_code(failed_batches: int) -> int:
    """0 when every lookup went through, EXIT_INCOMPLETE when some batches must be redone."""
    return EXIT_INCOMPLETE if failed_batches else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    settings = Settings()  # type: ignore[call-arg]  # values come from the environment
    with (
        httpx.Client(timeout=30.0, headers={"User-Agent": "graphrag-student-project"}) as client,
        Neo4jStore(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password.get_secret_value(),
        ) as store,
    ):
        if args.citations_only:
            store.ensure_schema()
            cites = run_citations(SemanticScholarSource(client), store, store.article_ids())
            print(cites, store.counts())
            return exit_code(cites.failed_batches)
        report = run_ingestion(
            ArxivSource(client),
            OpenAlexSource(client),
            SemanticScholarSource(client),
            store,
            args.since,
            args.until,
            limit=args.limit,
        )
    print(report)
    return exit_code(report.citation_batches_failed)


if __name__ == "__main__":
    raise SystemExit(main())
