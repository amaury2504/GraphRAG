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
from graphrag.ports.embedding import Embedder
from graphrag.ports.indexing import ChunkStore
from graphrag.settings import Settings

ARTICLES_PER_BATCH = 64


@dataclass(frozen=True, slots=True)
class IndexReport:
    articles: int
    chunks: int
    total_chunks: int
    removed: int = 0  # orphan chunks deleted from re-indexed articles


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
    removed = 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        indexed = index_articles(batch, embedder)
        # New chunks, orphans and hash of these articles all land in one transaction.
        removed += store.upsert_chunks(indexed, {a.arxiv_id: article_hash(a) for a in batch})
        written += len(indexed)
        progress = f"chunks: {start + len(batch)}/{len(todo)} articles indexed, {written} chunks"
        log(f"{progress}, {removed} removed" if removed else progress)
    return IndexReport(
        articles=len(todo), chunks=written, total_chunks=store.chunk_count(), removed=removed
    )


def stale_articles(states: Sequence[ArticleIndexState], limit: int | None = None) -> list[Article]:
    """Articles whose current hash differs from the one they were indexed from (or none)."""
    stale = [s.article for s in states if s.indexed_hash != article_hash(s.article)]
    return stale if limit is None else stale[:limit]


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed article chunks into Neo4j.")
    parser.add_argument("--limit", type=int, default=None, help="index at most N articles")
    parser.add_argument("--batch-size", type=int, default=ARTICLES_PER_BATCH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    settings = Settings()  # type: ignore[call-arg]  # values come from the environment
    log_line("loading the embedding model")
    embedder = load_default_embedder()
    with Neo4jStore(
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password.get_secret_value()
    ) as store:
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
