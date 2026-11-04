"""Chunk storage and indexes on the real Neo4j.

Uses only test articles (ids 9999.*), removed afterwards. The indexes themselves are the
real ones (384 dimensions, as for bge-small), so creating them here is harmless.
"""

import math
from collections.abc import Iterator

import neo4j
import pytest

from graphrag.adapters.neo4j_store import FULLTEXT_INDEX, VECTOR_INDEX, Neo4jStore
from graphrag.domain.models import Chunk, IndexedChunk
from tests.integration.test_neo4j_store import URI, USER, _cleanup, _password, art

pytestmark = pytest.mark.integration

DIMENSIONS = 384


@pytest.fixture
def store() -> Iterator[Neo4jStore]:
    _cleanup()
    with Neo4jStore(URI, USER, _password()) as s:
        s.ensure_schema()
        s.ensure_chunk_schema(DIMENSIONS)
        yield s
    _cleanup()


def unit_vector(hot: int) -> tuple[float, ...]:
    """A vector pointing along one axis, so the nearest neighbour is known exactly."""
    return tuple(1.0 if i == hot else 0.0 for i in range(DIMENSIONS))


def indexed(n: int, position: int, text: str, hot: int) -> IndexedChunk:
    chunk = Chunk(f"9999.{n:05d}", position, text)
    return IndexedChunk(chunk, f"T{n}. {text}", unit_vector(hot))


def test_chunk_schema_is_repeatable(store: Neo4jStore) -> None:
    store.ensure_chunk_schema(DIMENSIONS)


def test_vector_index_with_other_dimensions_is_refused(store: Neo4jStore) -> None:
    with pytest.raises(ValueError, match="has 384 dimensions, the embedder has 5"):
        store.ensure_chunk_schema(5)


def test_dimensions_must_be_a_positive_int(store: Neo4jStore) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        store.ensure_chunk_schema(0)


def test_chunks_are_linked_to_their_article_and_rewrites_add_nothing(store: Neo4jStore) -> None:
    store.upsert_articles([art(1), art(2)])
    before = store.chunk_count()
    chunks = [indexed(1, 0, "first part", 0), indexed(1, 1, "second part", 1)]
    store.upsert_chunks(chunks)
    store.upsert_chunks(chunks)
    assert store.chunk_count() - before == 2
    with neo4j.GraphDatabase.driver(URI, auth=(USER, _password())) as driver:
        records, _, _ = driver.execute_query(
            "MATCH (c:Chunk)-[:PART_OF]->(a:Article {arxiv_id: '9999.00001'}) "
            "RETURN c.key AS key, c.text AS text, c.search_text AS search, "
            "size(c.embedding) AS dims ORDER BY c.position"
        )
    assert [(r["key"], r["text"], r["search"], r["dims"]) for r in records] == [
        ("9999.00001#0", "first part", "T1. first part", DIMENSIONS),
        ("9999.00001#1", "second part", "T1. second part", DIMENSIONS),
    ]


def test_articles_without_chunks(store: Neo4jStore) -> None:
    store.upsert_articles([art(1), art(2)])
    store.upsert_chunks([indexed(1, 0, "text", 0)])
    todo = [a.arxiv_id for a in store.articles_without_chunks() if a.arxiv_id.startswith("9999.")]
    assert todo == ["9999.00002"]
    (article,) = [a for a in store.articles_without_chunks() if a.arxiv_id == "9999.00002"]
    assert article == art(2)


def test_articles_without_chunks_respects_limit(store: Neo4jStore) -> None:
    store.upsert_articles([art(1), art(2)])
    assert len(store.articles_without_chunks(limit=1)) == 1


def test_vector_index_finds_the_nearest_chunk(store: Neo4jStore) -> None:
    store.upsert_articles([art(1), art(2)])
    store.upsert_chunks([indexed(1, 0, "near", 7), indexed(2, 0, "far", 8)])
    with neo4j.GraphDatabase.driver(URI, auth=(USER, _password())) as driver:
        records, _, _ = driver.execute_query(
            "CALL db.index.vector.queryNodes($index, 50, $vector) YIELD node, score "
            "WHERE node.arxiv_id STARTS WITH '9999.' RETURN node.key AS key, score",
            index=VECTOR_INDEX,
            vector=list(unit_vector(7)),
        )
    assert records[0]["key"] == "9999.00001#0"
    assert math.isclose(records[0]["score"], 1.0, rel_tol=1e-6)


def test_fulltext_index_searches_title_and_chunk(store: Neo4jStore) -> None:
    store.upsert_articles([art(1)])
    store.upsert_chunks([indexed(1, 0, "zqxwv benchmark results", 0)])
    with neo4j.GraphDatabase.driver(URI, auth=(USER, _password())) as driver:
        by_text, _, _ = driver.execute_query(
            "CALL db.index.fulltext.queryNodes($index, 'zqxwv') YIELD node RETURN node.key AS key",
            index=FULLTEXT_INDEX,
        )
        by_title, _, _ = driver.execute_query(
            "CALL db.index.fulltext.queryNodes($index, 'T1') YIELD node "
            "WHERE node.arxiv_id STARTS WITH '9999.' RETURN node.key AS key",
            index=FULLTEXT_INDEX,
        )
    assert [r["key"] for r in by_text] == ["9999.00001#0"]
    assert [r["key"] for r in by_title] == ["9999.00001#0"]
