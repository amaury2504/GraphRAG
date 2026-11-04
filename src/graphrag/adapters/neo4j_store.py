from collections.abc import Iterator, Sequence
from itertools import islice
from typing import Any

import neo4j

from graphrag.domain.models import (
    Article,
    GraphCounts,
    IndexedChunk,
    ReferenceRecord,
    WorkRecord,
)

WRITE_BATCH = 500

SCHEMA = (
    "CREATE CONSTRAINT article_arxiv_id IF NOT EXISTS FOR (a:Article) REQUIRE a.arxiv_id IS UNIQUE",
    "CREATE CONSTRAINT author_key IF NOT EXISTS FOR (a:Author) REQUIRE a.key IS UNIQUE",
)

UPSERT_ARTICLES = """
UNWIND $rows AS row
MERGE (a:Article {arxiv_id: row.arxiv_id})
ON CREATE SET a.title = row.title,
              a.abstract = row.abstract,
              a.published = date(row.published),
              a.categories = row.categories,
              a.author_names = row.author_names
"""

UPSERT_WORKS = """
UNWIND $rows AS row
MATCH (a:Article {arxiv_id: row.arxiv_id})
SET a.openalex_id = row.openalex_id
WITH a, row
UNWIND row.authors AS author
MERGE (p:Author {key: author.key})
ON CREATE SET p.name = author.name
MERGE (a)-[r:AUTHORED_BY]->(p)
SET r.position = author.position
"""

UPSERT_CITATIONS = """
UNWIND $rows AS row
MATCH (s:Article {arxiv_id: row.arxiv_id})
UNWIND row.cited AS cited_id
MATCH (t:Article {arxiv_id: cited_id})
WHERE t <> s
MERGE (s)-[:CITES]->(t)
"""


VECTOR_INDEX = "chunk_embedding"
FULLTEXT_INDEX = "chunk_text"

CHUNK_KEY_CONSTRAINT = (
    "CREATE CONSTRAINT chunk_key IF NOT EXISTS FOR (c:Chunk) REQUIRE c.key IS UNIQUE"
)


def create_vector_index(dimensions: int) -> str:
    """Schema commands take the size as a literal; callers pass a checked int."""
    return f"""
CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS
FOR (c:Chunk) ON c.embedding
OPTIONS {{indexConfig: {{
  `vector.dimensions`: {dimensions},
  `vector.similarity_function`: 'cosine'
}}}}
"""


CREATE_FULLTEXT_INDEX = (
    f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX} IF NOT EXISTS FOR (c:Chunk) ON EACH [c.search_text]"
)
VECTOR_INDEX_OPTIONS = "SHOW VECTOR INDEXES YIELD name, options WHERE name = $name RETURN options"

ARTICLES_WITHOUT_CHUNKS = """
MATCH (a:Article)
WHERE NOT EXISTS { (:Chunk)-[:PART_OF]->(a) }
RETURN a ORDER BY a.arxiv_id
"""

UPSERT_CHUNKS = """
UNWIND $rows AS row
MATCH (a:Article {arxiv_id: row.arxiv_id})
MERGE (c:Chunk {key: row.key})
SET c.arxiv_id = row.arxiv_id,
    c.position = row.position,
    c.text = row.text,
    c.search_text = row.search_text,
    c.embedding = row.embedding
MERGE (c)-[:PART_OF]->(a)
"""


def _article(node: Any) -> Article:
    return Article(
        arxiv_id=node["arxiv_id"],
        title=node["title"],
        abstract=node["abstract"],
        published=node["published"].to_native(),
        categories=tuple(node.get("categories") or ()),
        author_names=tuple(node.get("author_names") or ()),
    )


def _chunks(rows: Sequence[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    iterator = iter(rows)
    while chunk := list(islice(iterator, size)):
        yield chunk


class Neo4jStore:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> Neo4jStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def ensure_schema(self) -> None:
        with self._driver.session() as session:
            for statement in SCHEMA:
                session.run(statement).consume()

    def upsert_articles(self, articles: Sequence[Article]) -> None:
        rows = [
            {
                "arxiv_id": a.arxiv_id,
                "title": a.title,
                "abstract": a.abstract,
                "published": a.published.isoformat(),
                "categories": list(a.categories),
                "author_names": list(a.author_names),
            }
            for a in articles
        ]
        self._write(UPSERT_ARTICLES, rows)

    def upsert_works(self, records: Sequence[WorkRecord]) -> None:
        rows = [
            {
                "arxiv_id": r.arxiv_id,
                "openalex_id": r.openalex_id,
                "authors": [
                    {"key": a.key, "name": a.name, "position": i} for i, a in enumerate(r.authors)
                ],
            }
            for r in records
        ]
        self._write(UPSERT_WORKS, rows)

    def upsert_citations(self, records: Sequence[ReferenceRecord]) -> None:
        rows = [{"arxiv_id": r.arxiv_id, "cited": list(r.cited_arxiv_ids)} for r in records]
        self._write(UPSERT_CITATIONS, rows)

    def article_ids(self) -> list[str]:
        with self._driver.session() as session:
            result = session.run("MATCH (a:Article) RETURN a.arxiv_id AS id ORDER BY id")
            return [str(record["id"]) for record in result]

    def counts(self) -> GraphCounts:
        query = """
        CALL () { MATCH (a:Article) RETURN count(a) AS articles }
        CALL () { MATCH (p:Author) RETURN count(p) AS authors }
        CALL () { MATCH ()-[r:AUTHORED_BY]->() RETURN count(r) AS authored_by }
        CALL () { MATCH ()-[r:CITES]->() RETURN count(r) AS cites }
        RETURN articles, authors, authored_by, cites
        """
        with self._driver.session() as session:
            record = session.run(query).single()
        if record is None:
            raise RuntimeError("count query returned no row")
        return GraphCounts(
            articles=record["articles"],
            authors=record["authors"],
            authored_by=record["authored_by"],
            cites=record["cites"],
        )

    def ensure_chunk_schema(self, dimensions: int) -> None:
        if type(dimensions) is not int or dimensions < 1:
            raise ValueError(f"dimensions must be a positive integer, got {dimensions!r}")
        with self._driver.session() as session:
            session.run(CHUNK_KEY_CONSTRAINT).consume()
            session.run(create_vector_index(dimensions)).consume()
            session.run(CREATE_FULLTEXT_INDEX).consume()
            record = session.run(VECTOR_INDEX_OPTIONS, name=VECTOR_INDEX).single()
        if record is None:
            raise RuntimeError(f"vector index {VECTOR_INDEX} was not created")
        existing = record["options"]["indexConfig"]["vector.dimensions"]
        if existing != dimensions:
            raise ValueError(
                f"vector index {VECTOR_INDEX} has {existing} dimensions, the embedder has "
                f"{dimensions}: drop the index and the chunks, then index again"
            )

    def articles_without_chunks(self, limit: int | None = None) -> list[Article]:
        query = ARTICLES_WITHOUT_CHUNKS + (" LIMIT $limit" if limit is not None else "")
        with self._driver.session() as session:
            return [_article(record["a"]) for record in session.run(query, limit=limit)]

    def upsert_chunks(self, chunks: Sequence[IndexedChunk]) -> None:
        rows = [
            {
                "key": c.key,
                "arxiv_id": c.chunk.arxiv_id,
                "position": c.chunk.position,
                "text": c.chunk.text,
                "search_text": c.search_text,
                "embedding": list(c.embedding),
            }
            for c in chunks
        ]
        if not rows:
            return

        def write(tx: neo4j.ManagedTransaction) -> None:
            tx.run(UPSERT_CHUNKS, rows=rows).consume()

        with self._driver.session() as session:
            session.execute_write(write)  # one transaction: an article is never half indexed

    def chunk_count(self) -> int:
        with self._driver.session() as session:
            record = session.run("MATCH (c:Chunk) RETURN count(c) AS n").single()
        return 0 if record is None else int(record["n"])

    def _write(self, query: str, rows: Sequence[dict[str, Any]]) -> None:
        def run_chunk(tx: neo4j.ManagedTransaction, chunk: list[dict[str, Any]]) -> None:
            tx.run(query, rows=chunk).consume()

        with self._driver.session() as session:
            for chunk in _chunks(rows, WRITE_BATCH):
                session.execute_write(run_chunk, chunk)
