import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import islice

import httpx

from graphrag.adapters.arxiv import ArxivSource
from graphrag.adapters.neo4j_store import Neo4jStore
from graphrag.adapters.openalex import BATCH_SIZE, OpenAlexSource
from graphrag.adapters.semantic_scholar import BATCH_SIZE as SEMANTIC_SCHOLAR_BATCH
from graphrag.adapters.semantic_scholar import SemanticScholarSource
from graphrag.domain.models import Article, GraphCounts, ReferenceRecord, WorkRecord
from graphrag.ports.ingestion import AuthorSource, CitationSource, GraphStore, PaperSource
from graphrag.settings import Settings

ARTICLE_BATCH = 100


@dataclass(frozen=True, slots=True)
class IngestionReport:
    articles_fetched: int
    authors_found: int
    authors_missing: int
    references_found: int
    references_missing: int
    without_arxiv_references: int
    counts: GraphCounts


def run_ingestion(
    papers: PaperSource,
    authors: AuthorSource,
    citations: CitationSource,
    store: GraphStore,
    start: date,
    end: date,
    *,
    limit: int | None = None,
    log: Callable[[str], None] = print,
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

    # Citations are written once every article is stored, so targets always exist.
    references: list[ReferenceRecord] = []
    for offset in range(0, len(ids), SEMANTIC_SCHOLAR_BATCH):
        found_refs = citations.fetch_references(ids[offset : offset + SEMANTIC_SCHOLAR_BATCH])
        references.extend(found_refs)
        done = min(offset + SEMANTIC_SCHOLAR_BATCH, len(ids))
        log(f"references (semantic scholar): {done}/{len(ids)} looked up")
    store.upsert_citations(references)

    return IngestionReport(
        articles_fetched=fetched,
        authors_found=len(works),
        authors_missing=len(ids) - len(works),
        references_found=len(references),
        references_missing=len(ids) - len(references),
        without_arxiv_references=sum(1 for r in references if not r.cited_arxiv_ids),
        counts=store.counts(),
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    today = date.today()
    parser = argparse.ArgumentParser(description="Ingest arXiv RAG papers into Neo4j.")
    parser.add_argument("--since", type=date.fromisoformat, default=today - timedelta(days=365))
    parser.add_argument("--until", type=date.fromisoformat, default=today)
    parser.add_argument("--limit", type=int, default=None, help="stop after N articles (test lot)")
    return parser.parse_args(argv)


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
