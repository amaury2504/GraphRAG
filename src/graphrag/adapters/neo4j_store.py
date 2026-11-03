from collections.abc import Iterator, Sequence
from itertools import islice
from typing import Any

import neo4j

from graphrag.domain.models import Article, GraphCounts, WorkRecord

WRITE_BATCH = 500

SCHEMA = (
    "CREATE CONSTRAINT article_arxiv_id IF NOT EXISTS FOR (a:Article) REQUIRE a.arxiv_id IS UNIQUE",
    "CREATE CONSTRAINT author_key IF NOT EXISTS FOR (a:Author) REQUIRE a.key IS UNIQUE",
    "CREATE INDEX article_openalex_id IF NOT EXISTS FOR (a:Article) ON (a.openalex_id)",
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
UNWIND row.refs AS ref
MATCH (t:Article {openalex_id: ref})
WHERE t <> s
MERGE (s)-[:CITES]->(t)
"""


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

    def upsert_citations(self, records: Sequence[WorkRecord]) -> None:
        rows = [{"arxiv_id": r.arxiv_id, "refs": list(r.referenced_openalex_ids)} for r in records]
        self._write(UPSERT_CITATIONS, rows)

    def article_ids(self) -> list[str]:
        with self._driver.session() as session:
            result = session.run("MATCH (a:Article) RETURN a.arxiv_id AS id ORDER BY id")
            return [str(record["id"]) for record in result]

    def counts(self) -> GraphCounts:
        query = """
        CALL { MATCH (a:Article) RETURN count(a) AS articles }
        CALL { MATCH (p:Author) RETURN count(p) AS authors }
        CALL { MATCH ()-[r:AUTHORED_BY]->() RETURN count(r) AS authored_by }
        CALL { MATCH ()-[r:CITES]->() RETURN count(r) AS cites }
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

    def _write(self, query: str, rows: Sequence[dict[str, Any]]) -> None:
        def run_chunk(tx: neo4j.ManagedTransaction, chunk: list[dict[str, Any]]) -> None:
            tx.run(query, rows=chunk).consume()

        with self._driver.session() as session:
            for chunk in _chunks(rows, WRITE_BATCH):
                session.execute_write(run_chunk, chunk)
