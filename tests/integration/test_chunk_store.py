"""Chunk storage and indexes on the real Neo4j.

Uses only test articles (ids 9999.*), removed afterwards. The indexes themselves are the
real ones (384 dimensions, as for bge-small), so creating them here is harmless.
"""

import math
from collections.abc import Iterator

import neo4j
import pytest

from graphrag.adapters.neo4j_store import (
    FULLTEXT_ANALYZER,
    FULLTEXT_INDEX,
    VECTOR_INDEX,
    Neo4jStore,
)
from graphrag.domain.models import Chunk, IndexedChunk, IndexInfo
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


def test_vector_search_returns_hits_with_their_article(store: Neo4jStore) -> None:
    store.upsert_articles([art(1), art(2)])
    store.upsert_chunks([indexed(1, 0, "near", 7), indexed(2, 0, "far", 8)])
    hits = [h for h in store.vector_search(unit_vector(7), 50) if h.arxiv_id.startswith("9999.")]
    first = hits[0]
    assert (first.arxiv_id, first.title, first.position, first.text) == (
        "9999.00001",
        "T1",
        0,
        "near",
    )
    assert math.isclose(first.score, 1.0, rel_tol=1e-6)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_keyword_search_matches_words_and_respects_k(store: Neo4jStore) -> None:
    store.upsert_articles([art(1), art(2)])
    store.upsert_chunks([indexed(1, 0, "zqxwv zqxwv retrieval", 0), indexed(2, 0, "zqxwv once", 1)])
    hits = store.keyword_search("ZQXWV", 10)
    assert [h.arxiv_id for h in hits] == ["9999.00001", "9999.00002"]
    assert hits[0].text == "zqxwv zqxwv retrieval"
    assert len(store.keyword_search("zqxwv", 1)) == 1


@pytest.mark.parametrize(
    "question",
    [
        'what is "zqxwv"? (AND/OR) a+b -c ~ * : \\ / [x TO y] {z} ^2 && || !',
        "zqxwv AND NOT",
        "zqxwv-based: retrieval?",
    ],
)
def test_keyword_search_accepts_any_user_text(store: Neo4jStore, question: str) -> None:
    store.upsert_articles([art(1)])
    store.upsert_chunks([indexed(1, 0, "zqxwv retrieval", 0)])
    assert "9999.00001" in [h.arxiv_id for h in store.keyword_search(question, 10)]


@pytest.mark.parametrize("question", ["", "   ", "???", '"" ()'])
def test_keyword_search_without_words_finds_nothing(store: Neo4jStore, question: str) -> None:
    assert store.keyword_search(question, 10) == []


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


def _analyzer() -> str:
    with neo4j.GraphDatabase.driver(URI, auth=(USER, _password())) as driver:
        records, _, _ = driver.execute_query(
            "SHOW FULLTEXT INDEXES YIELD name, options WHERE name = $name RETURN options",
            name=FULLTEXT_INDEX,
        )
    return str(records[0]["options"]["indexConfig"]["fulltext.analyzer"])


def test_keyword_search_matches_word_forms_and_ignores_stop_words(store: Neo4jStore) -> None:
    store.upsert_articles([art(1)])
    store.upsert_chunks([indexed(1, 0, "zqxwv autonomous agent", 0)])
    assert _analyzer() == FULLTEXT_ANALYZER
    assert "9999.00001" in [h.arxiv_id for h in store.keyword_search("zqxwv agents", 10)]
    assert store.keyword_search("the of and", 10) == []


def test_an_index_with_another_analyzer_is_rebuilt(store: Neo4jStore) -> None:
    store.upsert_articles([art(1)])
    store.upsert_chunks([indexed(1, 0, "zqxwv agent", 0)])
    with neo4j.GraphDatabase.driver(URI, auth=(USER, _password())) as driver:
        driver.execute_query(f"DROP INDEX {FULLTEXT_INDEX}")
        driver.execute_query(
            f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX} FOR (c:Chunk) ON EACH [c.search_text] "
            "OPTIONS {indexConfig: {`fulltext.analyzer`: 'standard-no-stop-words'}}"
        )
    assert _analyzer() == "standard-no-stop-words"
    store.ensure_chunk_schema(DIMENSIONS)
    assert _analyzer() == FULLTEXT_ANALYZER
    assert "9999.00001" in [h.arxiv_id for h in store.keyword_search("zqxwv agents", 10)]


def test_searches_return_the_whole_abstract_of_the_article(store: Neo4jStore) -> None:
    article = art(1)
    store.upsert_articles([article])
    store.upsert_chunks([indexed(1, 0, "zqxwv first part", 0), indexed(1, 1, "second part", 1)])
    by_meaning = store.vector_search(unit_vector(0), 5)
    by_keywords = store.keyword_search("zqxwv", 5)
    for hits in (by_meaning, by_keywords):
        mine = [h for h in hits if h.arxiv_id == article.arxiv_id]
        assert mine
        assert all(h.abstract == article.abstract for h in mine)
    assert by_keywords[0].text == "zqxwv first part"  # still the matching chunk


INFO = IndexInfo("test-model", "rev1", DIMENSIONS, "title. chunk")


@pytest.fixture
def clean_info(store: Neo4jStore) -> Iterator[Neo4jStore]:
    """Run with no fingerprint, then put back the real one."""
    saved = store.read_index_info()
    with neo4j.GraphDatabase.driver(URI, auth=(USER, _password())) as driver:
        driver.execute_query("MATCH (i:IndexInfo {name: 'chunks'}) DELETE i")
    yield store
    with neo4j.GraphDatabase.driver(URI, auth=(USER, _password())) as driver:
        driver.execute_query("MATCH (i:IndexInfo {name: 'chunks'}) DELETE i")
    if saved is not None:
        store.ensure_index_info(saved)


def test_the_first_fingerprint_is_recorded_and_read_back(clean_info: Neo4jStore) -> None:
    assert clean_info.read_index_info() is None
    clean_info.ensure_index_info(INFO)
    assert clean_info.read_index_info() == INFO
    clean_info.ensure_index_info(INFO)  # repeatable
    clean_info.check_index_info(INFO)


def test_another_model_with_the_same_dimensions_is_refused(clean_info: Neo4jStore) -> None:
    clean_info.ensure_index_info(INFO)
    other = IndexInfo("other-model", "rev1", DIMENSIONS, "title. chunk")
    with pytest.raises(ValueError, match="model: stored 'test-model', current 'other-model'"):
        clean_info.ensure_index_info(other)
    with pytest.raises(ValueError, match="built differently"):
        clean_info.check_index_info(other)
    assert clean_info.read_index_info() == INFO  # a refusal changes nothing


def test_checking_without_a_recorded_fingerprint_fails(clean_info: Neo4jStore) -> None:
    with pytest.raises(ValueError, match="no index fingerprint"):
        clean_info.check_index_info(INFO)
