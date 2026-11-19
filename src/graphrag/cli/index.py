"""Cut abstracts into chunks, embed them and store them in Neo4j with their indexes.

Only articles whose hash (title, abstract, chunker version) differs from the one they were
indexed from are processed: a rerun resumes where the last one stopped, picks up new and
modified articles, and leaves the others untouched.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from graphrag.adapters.neo4j_store import Neo4jStore
from graphrag.adapters.sentence_transformer import default_index_info, load_default_embedder
from graphrag.cli.ingest import log_line
from graphrag.domain.chunking import chunk_abstract, search_text
from graphrag.domain.hashing import article_hash
from graphrag.domain.models import Article, ArticleIndexState, IndexedChunk, IndexInfo
from graphrag.domain.versioning import GcReport, VersionInfo
from graphrag.ports.embedding import Embedder
from graphrag.ports.indexing import ChunkStore
from graphrag.settings import Settings

ARTICLES_PER_BATCH = 64
DEFAULT_KEPT_VERSIONS = 3


@dataclass(frozen=True, slots=True)
class IndexReport:
    articles: int
    chunks: int
    total_chunks: int  # every chunk stored, including those no current version contains
    version: int | None = None  # the version published by this run, None if it published none


def index_articles(articles: Sequence[Article], embedder: Embedder) -> list[IndexedChunk]:
    chunks = [c for a in articles for c in chunk_abstract(a.arxiv_id, a.abstract)]
    titles = {a.arxiv_id: a.title for a in articles}
    texts = [search_text(titles[c.arxiv_id], c.text) for c in chunks]
    vectors = embedder.embed_documents(texts)
    return [
        IndexedChunk(chunk, text, tuple(vector))
        for chunk, text, vector in zip(chunks, texts, vectors, strict=True)
    ]


def run_indexing(
    store: ChunkStore,
    embedder: Embedder,
    *,
    limit: int | None = None,
    batch_size: int = ARTICLES_PER_BATCH,
    log: Callable[[str], None] = log_line,
    info: IndexInfo | None = None,
) -> IndexReport:
    store.ensure_chunk_schema(embedder.dimensions)
    if info is not None:
        store.ensure_index_info(info)
    todo = stale_articles(store.articles_with_index_state(), limit)
    log(f"chunks: {len(todo)} articles to index")
    written = 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        indexed = index_articles(batch, embedder)
        # New chunks and the hash of these articles land in one transaction; readers do not
        # see them before the version is published.
        store.upsert_chunks(indexed, {a.arxiv_id: article_hash(a) for a in batch})
        written += len(indexed)
        log(f"chunks: {start + len(batch)}/{len(todo)} articles indexed, {written} chunks")
    version: VersionInfo | None = None
    if todo or store.current_version() is None:  # also adopts chunks indexed before versions
        version = store.publish_version()
        log(f"chunks: version {version.id} published ({version.chunk_count} chunks)")
    return IndexReport(
        articles=len(todo),
        chunks=written,
        total_chunks=store.chunk_count(),
        version=None if version is None else version.id,
    )


def run_rollback(store: ChunkStore, *, log: Callable[[str], None] = log_line) -> VersionInfo:
    """Make the previous version current again (the abandoned one's chunks stay until a gc)."""
    version = store.rollback()
    log(f"chunks: rolled back to version {version.id} ({version.chunk_count} chunks)")
    return version


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def run_gc(
    store: ChunkStore,
    *,
    keep: int,
    min_age_days: int,
    apply: bool,
    log: Callable[[str], None] = log_line,
) -> GcReport:
    """Drop old versions and the chunks no kept version references (a dry run unless `apply`)."""
    report = store.collect_garbage(keep=keep, min_age_days=min_age_days, apply=apply)
    kept = ", ".join(str(v) for v in report.kept_versions)
    log(f"gc: versions kept {kept}" if apply else f"gc: dry run, versions kept {kept}")
    if not report.dropped_versions and not report.unreferenced_chunks:
        log("gc: nothing to remove")
        return report
    dropped = ", ".join(str(v) for v in report.dropped_versions)
    what = (
        f"{_plural(len(report.dropped_versions), 'version')} ({dropped}) "
        f"and {_plural(report.unreferenced_chunks, 'chunk')}"
    )
    log(f"gc: dropped {what}" if apply else f"gc: would drop {what}, rerun with --apply to delete")
    return report


def stale_articles(states: Sequence[ArticleIndexState], limit: int | None = None) -> list[Article]:
    """Articles whose current hash differs from the one they were indexed from (or none)."""
    stale = [s.article for s in states if s.indexed_hash != article_hash(s.article)]
    return stale if limit is None else stale[:limit]


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed article chunks into Neo4j.")
    parser.add_argument("--limit", type=int, default=None, help="index at most N articles")
    parser.add_argument("--batch-size", type=int, default=ARTICLES_PER_BATCH)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--rollback",
        action="store_true",
        help="make the previous index version current again, then stop (no indexing)",
    )
    mode.add_argument(
        "--gc",
        action="store_true",
        help="garbage-collect old versions and unreferenced chunks (dry run unless --apply)",
    )
    parser.add_argument(
        "--keep", type=int, default=DEFAULT_KEPT_VERSIONS, help="versions to keep (--gc, min 2)"
    )
    parser.add_argument(
        "--min-age-days",
        type=int,
        default=0,
        help="also keep any version younger than this many days (--gc)",
    )
    parser.add_argument("--apply", action="store_true", help="really delete (--gc)")
    args = parser.parse_args(argv)
    gc_only = (args.apply, args.keep != DEFAULT_KEPT_VERSIONS, args.min_age_days != 0)
    if any(gc_only) and not args.gc:
        parser.error("--apply, --keep and --min-age-days only go with --gc")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    settings = Settings()  # type: ignore[call-arg]  # values come from the environment
    uri, user = settings.neo4j_uri, settings.neo4j_user
    password = settings.neo4j_password.get_secret_value()
    if args.rollback:
        with Neo4jStore(uri, user, password) as store:
            try:
                run_rollback(store)
            except LookupError as error:
                log_line(f"chunks: {error}")
                return 2
        return 0
    if args.gc:
        with Neo4jStore(uri, user, password) as store:
            try:
                run_gc(store, keep=args.keep, min_age_days=args.min_age_days, apply=args.apply)
            except (LookupError, ValueError) as error:
                log_line(f"gc: {error}")
                return 2
        return 0
    log_line("loading the embedding model")
    embedder = load_default_embedder()
    with Neo4jStore(uri, user, password) as store:
        report = run_indexing(
            store,
            embedder,
            limit=args.limit,
            batch_size=args.batch_size,
            info=default_index_info(embedder.dimensions),
        )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
