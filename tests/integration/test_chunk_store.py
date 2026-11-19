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
from graphrag.domain.hashing import article_hash
from graphrag.domain.models import ArticleIndexState, Chunk, IndexedChunk, IndexInfo
from tests.integration.test_neo4j_store import URI, USER, _cleanup, _password, _stored, art

pytestmark = pytest.mark.integration

DIMENSIONS = 384


TEST_POINTER = "test-index"


def _max_version_id() -> int:
    records = _stored("MATCH (v:Version) RETURN coalesce(max(v.id), 0) AS m")
    return int(records[0]["m"])


def _drop_test_versions(above: int) -> None:
    with neo4j.GraphDatabase.driver(URI, auth=(USER, _password())) as driver:
        driver.execute_query("MATCH (c:Current {name: $name}) DETACH DELETE c", name=TEST_POINTER)
        driver.execute_query("MATCH (v:Version) WHERE v.id > $above DETACH DELETE v", above=above)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Iterator[Neo4jStore]:
    """The real database, but versions use their own pointer and are removed afterwards."""
    monkeypatch.setattr("graphrag.adapters.neo4j_store.CURRENT_NAME", TEST_POINTER)
    _cleanup()
    with Neo4jStore(URI, USER, _password()) as s:
        s.ensure_schema()
        s.ensure_chunk_schema(DIMENSIONS)
        above = _max_version_id()
        yield s
        _drop_test_versions(above)
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
        (chunks[0].key, "first part", "T1. first part", DIMENSIONS),
        (chunks[1].key, "second part", "T1. second part", DIMENSIONS),
    ]


def ours(states: list[ArticleIndexState]) -> dict[str, ArticleIndexState]:
    return {s.article.arxiv_id: s for s in states if s.article.arxiv_id.startswith("9999.")}


def test_index_state_lists_articles_with_the_hash_they_were_indexed_from(
    store: Neo4jStore,
) -> None:
    store.upsert_articles([art(1), art(2)])
    store.upsert_chunks([indexed(1, 0, "text", 0)], {"9999.00001": "h1"})
    states = ours(store.articles_with_index_state())
    assert list(states) == ["9999.00001", "9999.00002"]  # ordered by id
    assert states["9999.00001"].indexed_hash == "h1"
    assert states["9999.00002"].indexed_hash is None
    assert states["9999.00002"].article == art(2)


def test_chunks_written_without_a_hash_leave_the_article_unindexed(store: Neo4jStore) -> None:
    store.upsert_articles([art(1)])
    store.upsert_chunks([indexed(1, 0, "text", 0)])
    assert ours(store.articles_with_index_state())["9999.00001"].indexed_hash is None


def keys_of(article_id: str) -> list[str]:
    records = _stored(
        "MATCH (c:Chunk)-[:PART_OF]->(:Article {arxiv_id: $id}) RETURN c.key AS k ORDER BY k",
        id=article_id,
    )
    return [r["k"] for r in records]


def owned_keys(article_id: str) -> list[str]:
    records = _stored(
        "MATCH (a:Article {arxiv_id: $id}) RETURN coalesce(a.indexed_keys, []) AS k", id=article_id
    )
    return sorted(records[0]["k"])


def test_replacing_an_article_keeps_the_old_chunks_and_adds_the_new_ones(
    store: Neo4jStore,
) -> None:
    store.upsert_articles([art(1)])
    three = [indexed(1, p, f"part {p}", p) for p in range(3)]
    store.upsert_chunks(three, {"9999.00001": "h1"})
    only = indexed(1, 0, "only part", 0)
    store.upsert_chunks([only], {"9999.00001": "h2"})
    assert len(keys_of("9999.00001")) == 4  # nothing deleted: that is the garbage collector's job
    assert owned_keys("9999.00001") == [only.key]  # but only the new chunk is owned
    assert ours(store.articles_with_index_state())["9999.00001"].indexed_hash == "h2"


def test_a_chunk_that_already_exists_is_not_rewritten(store: Neo4jStore) -> None:
    store.upsert_articles([art(1)])
    first = indexed(1, 0, "same words", 0)
    store.upsert_chunks([first], {"9999.00001": "h1"})
    same_key_other_vector = IndexedChunk(first.chunk, first.search_text, unit_vector(9))
    store.upsert_chunks([same_key_other_vector], {"9999.00001": "h2"})
    records = _stored("MATCH (c:Chunk {key: $key}) RETURN c.embedding[0] AS first", key=first.key)
    assert len(records) == 1
    assert records[0]["first"] == 1.0  # still the original vector (hot axis 0)


def test_an_article_emptied_by_a_replacement_owns_no_chunk(store: Neo4jStore) -> None:
    store.upsert_articles([art(1)])
    store.upsert_chunks([indexed(1, 0, "a", 0), indexed(1, 1, "b", 1)], {"9999.00001": "h1"})
    store.upsert_chunks([], {"9999.00001": "h2"})
    assert owned_keys("9999.00001") == []
    assert len(keys_of("9999.00001")) == 2
    assert ours(store.articles_with_index_state())["9999.00001"].indexed_hash == "h2"


def test_replacing_one_article_leaves_the_others_alone(store: Neo4jStore) -> None:
    store.upsert_articles([art(1), art(2)])
    second = [indexed(2, 0, "c", 2), indexed(2, 1, "d", 3)]
    store.upsert_chunks(
        [indexed(1, 0, "a", 0), indexed(1, 1, "b", 1), *second],
        {"9999.00001": "h1", "9999.00002": "h2"},
    )
    store.upsert_chunks([indexed(1, 0, "a", 0)], {"9999.00001": "h1b"})
    assert owned_keys("9999.00002") == sorted(c.key for c in second)
    assert ours(store.articles_with_index_state())["9999.00002"].indexed_hash == "h2"


def test_a_failed_write_changes_nothing(store: Neo4jStore, monkeypatch: pytest.MonkeyPatch) -> None:
    store.upsert_articles([art(1)])
    old = [indexed(1, 0, "a", 0), indexed(1, 1, "b", 1)]
    store.upsert_chunks(old, {"9999.00001": "h1"})
    # The last statement of the transaction fails, after the new chunk was written: the
    # chunk, the owned keys and the hash must all be rolled back.
    monkeypatch.setattr("graphrag.adapters.neo4j_store.MARK_INDEXED", "NOT CYPHER")
    with pytest.raises(neo4j.exceptions.Neo4jError):
        store.upsert_chunks([indexed(1, 0, "rewritten", 5)], {"9999.00001": "h2"})
    assert keys_of("9999.00001") == sorted(c.key for c in old)
    assert owned_keys("9999.00001") == sorted(c.key for c in old)
    assert ours(store.articles_with_index_state())["9999.00001"].indexed_hash == "h1"


def test_no_rows_and_no_hashes_is_a_noop(store: Neo4jStore) -> None:
    store.upsert_chunks([])
    store.upsert_chunks([], {})
    assert keys_of("9999.00001") == []


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
    assert records[0]["key"] == indexed(1, 0, "near", 7).key
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
    key = indexed(1, 0, "zqxwv benchmark results", 0).key
    assert [r["key"] for r in by_text] == [key]
    assert [r["key"] for r in by_title] == [key]


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


def readable(store: Neo4jStore, hot: int) -> list[str]:
    """Texts of our test chunks that a vector search can reach, along axis `hot`."""
    hits = store.vector_search(unit_vector(hot), 50)
    return [h.text for h in hits if h.arxiv_id.startswith("9999.")]


def manifest(version_id: int) -> list[str]:
    records = _stored(
        "MATCH (:Version {id: $id})-[:CONTAINS]->(c:Chunk) "
        "WHERE c.arxiv_id STARTS WITH '9999.' RETURN c.key AS key ORDER BY key",
        id=version_id,
    )
    return [r["key"] for r in records]


def test_without_a_published_version_searches_read_every_chunk(store: Neo4jStore) -> None:
    store.upsert_articles([art(1)])
    store.upsert_chunks([indexed(1, 0, "legacy", 3)], {"9999.00001": "h1"})
    assert store.current_version() is None
    assert readable(store, 3) == ["legacy"]


def test_a_published_version_holds_the_chunks_of_the_indexed_articles(store: Neo4jStore) -> None:
    store.upsert_articles([art(1), art(2)])
    a, b = indexed(1, 0, "alpha", 0), indexed(2, 0, "beta", 1)
    store.upsert_chunks([a, b], {"9999.00001": "h1", "9999.00002": "h2"})
    version = store.publish_version()
    assert store.current_version() == version
    assert version.chunk_count >= 2
    assert manifest(version.id) == sorted([a.key, b.key])


def test_chunks_written_but_not_published_are_invisible_once_a_version_exists(
    store: Neo4jStore,
) -> None:
    store.upsert_articles([art(1), art(2)])
    store.upsert_chunks([indexed(1, 0, "published", 4)], {"9999.00001": "h1"})
    store.publish_version()
    store.upsert_chunks([indexed(2, 0, "pending", 4)], {"9999.00002": "h2"})
    assert readable(store, 4) == ["published"]
    store.publish_version()
    assert sorted(readable(store, 4)) == ["pending", "published"]


def test_a_new_version_shares_the_unchanged_chunks_and_drops_the_replaced_ones(
    store: Neo4jStore,
) -> None:
    store.upsert_articles([art(1), art(2)])
    kept, old = indexed(2, 0, "kept", 1), indexed(1, 0, "old", 0)
    store.upsert_chunks([old, kept], {"9999.00001": "h1", "9999.00002": "h2"})
    first = store.publish_version()
    new = indexed(1, 0, "new", 0)
    store.upsert_chunks([new], {"9999.00001": "h1b"})
    second = store.publish_version()
    assert second.id == first.id + 1
    assert manifest(first.id) == sorted([old.key, kept.key])
    assert manifest(second.id) == sorted([new.key, kept.key])
    assert readable(store, 0) == ["new"]  # the replaced chunk is stored but not readable
    assert len(keys_of("9999.00002")) == 1  # the shared chunk exists once


def test_the_pointer_moves_in_one_transaction_with_the_manifest(
    store: Neo4jStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.upsert_articles([art(1)])
    store.upsert_chunks([indexed(1, 0, "stable", 5)], {"9999.00001": "h1"})
    first = store.publish_version()
    store.upsert_chunks([indexed(1, 0, "next", 5)], {"9999.00001": "h2"})
    monkeypatch.setattr("graphrag.adapters.neo4j_store.POINT_TO_VERSION", "NOT CYPHER")
    with pytest.raises(neo4j.exceptions.Neo4jError):
        store.publish_version()  # fails after the manifest was built, on the pointer move
    assert store.current_version() == first
    assert readable(store, 5) == ["stable"]
    assert _max_version_id() == first.id  # the half-built version was rolled back too


def test_a_database_indexed_before_versions_existed_is_adopted_by_the_first_publish(
    store: Neo4jStore,
) -> None:
    store.upsert_articles([art(1)])
    legacy = IndexedChunk(
        Chunk("9999.00001", 0, "legacy words"), "T1. legacy words", unit_vector(6)
    )
    store.upsert_chunks([legacy], {"9999.00001": "h1"})
    _stored("MATCH (a:Article {arxiv_id: '9999.00001'}) REMOVE a.indexed_keys")
    version = store.publish_version()
    assert manifest(version.id) == [legacy.key]
    assert owned_keys("9999.00001") == [legacy.key]


def test_publishing_twice_without_change_gives_two_versions_with_the_same_chunks(
    store: Neo4jStore,
) -> None:
    store.upsert_articles([art(1)])
    store.upsert_chunks([indexed(1, 0, "same", 0)], {"9999.00001": "h1"})
    first, second = store.publish_version(), store.publish_version()
    assert second.id == first.id + 1
    assert manifest(first.id) == manifest(second.id)


def test_a_rollback_makes_the_previous_version_current_again(store: Neo4jStore) -> None:
    store.upsert_articles([art(1)])
    store.upsert_chunks([indexed(1, 0, "before", 7)], {"9999.00001": "h1"})
    first = store.publish_version()
    store.upsert_chunks([indexed(1, 0, "after", 7)], {"9999.00001": "h2"})
    store.publish_version()
    assert readable(store, 7) == ["after"]
    rolled_back = store.rollback()
    assert rolled_back.id == first.id
    assert store.current_version() == rolled_back
    assert readable(store, 7) == ["before"]
    assert len(keys_of("9999.00001")) == 2  # the abandoned chunk is kept for the GC


def test_a_rollback_restores_what_each_article_contributes_to_the_next_version(
    store: Neo4jStore,
) -> None:
    store.upsert_articles([art(1), art(2)])
    old = indexed(1, 0, "old", 0)
    store.upsert_chunks([old], {"9999.00001": "h1"})
    first = store.publish_version()
    store.upsert_chunks([indexed(1, 0, "new", 0)], {"9999.00001": "h1b"})
    store.upsert_chunks([indexed(2, 0, "late", 1)], {"9999.00002": "h2"})
    store.publish_version()
    store.rollback()
    assert owned_keys("9999.00001") == [old.key]
    assert owned_keys("9999.00002") == []  # it was not in the version rolled back to
    third = store.publish_version()  # a later publish does not bring the abandoned chunks back
    assert manifest(third.id) == manifest(first.id)
    assert third.id > first.id + 1


def test_a_rollback_can_be_repeated_back_through_the_history(store: Neo4jStore) -> None:
    store.upsert_articles([art(1)])
    ids = []
    for n in range(3):
        store.upsert_chunks([indexed(1, 0, f"state {n}", 0)], {"9999.00001": f"h{n}"})
        ids.append(store.publish_version().id)
    assert store.rollback().id == ids[1]
    assert store.rollback().id == ids[0]
    with pytest.raises(LookupError, match="no earlier version"):
        store.rollback()
    assert store.current_version() is not None
    assert store.current_version().id == ids[0]  # type: ignore[union-attr]


def test_a_rollback_without_any_version_fails(store: Neo4jStore) -> None:
    with pytest.raises(LookupError, match="no earlier version"):
        store.rollback()


def test_a_failed_rollback_changes_nothing(
    store: Neo4jStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.upsert_articles([art(1)])
    store.upsert_chunks([indexed(1, 0, "first", 8)], {"9999.00001": "h1"})
    store.publish_version()
    store.upsert_chunks([indexed(1, 0, "second", 8)], {"9999.00001": "h2"})
    current = store.publish_version()
    monkeypatch.setattr("graphrag.adapters.neo4j_store.RESTORE_ARTICLE_KEYS", "NOT CYPHER")
    with pytest.raises(neo4j.exceptions.Neo4jError):
        store.rollback()
    assert store.current_version() == current  # the pointer move was rolled back
    assert readable(store, 8) == ["second"]


def test_searches_still_return_the_requested_number_of_current_chunks(store: Neo4jStore) -> None:
    store.upsert_articles([art(1)])
    for n in range(4):  # four stale chunks closer to the query than the readable one
        store.upsert_chunks([indexed(1, 0, f"stale {n}", 2)], {"9999.00001": f"h{n}"})
    store.publish_version()
    store.upsert_chunks([indexed(1, 0, "current", 2)], {"9999.00001": "hz"})
    store.publish_version()
    hits = [h for h in store.vector_search(unit_vector(2), 1) if h.arxiv_id.startswith("9999.")]
    assert [h.text for h in hits] == ["current"]


def test_keyword_search_reads_the_current_version_only(store: Neo4jStore) -> None:
    store.upsert_articles([art(1)])
    store.upsert_chunks([indexed(1, 0, "zqxwv oldword", 0)], {"9999.00001": "h1"})
    store.publish_version()
    store.upsert_chunks([indexed(1, 0, "zqxwv newword", 0)], {"9999.00001": "h2"})
    assert [h.text for h in store.keyword_search("zqxwv", 10)] == ["zqxwv oldword"]
    store.publish_version()
    assert [h.text for h in store.keyword_search("zqxwv", 10)] == ["zqxwv newword"]


def test_the_probe_counts_articles_without_chunks(store: Neo4jStore) -> None:
    from graphrag.adapters.health import Neo4jProbe

    probe = Neo4jProbe(URI, USER, _password())
    before = probe.unindexed_articles()
    store.upsert_articles([art(1), art(2)])
    assert probe.unindexed_articles() == before + 2
    store.upsert_chunks([indexed(1, 0, "some text", 0)], {"9999.00001": article_hash(art(1))})
    assert probe.unindexed_articles() == before + 1


def test_the_probe_counts_an_article_modified_after_indexing(store: Neo4jStore) -> None:
    from graphrag.adapters.health import Neo4jProbe

    probe = Neo4jProbe(URI, USER, _password())
    store.upsert_articles([art(1)])
    store.upsert_chunks([indexed(1, 0, "some text", 0)], {"9999.00001": article_hash(art(1))})
    before = probe.unindexed_articles()
    store.upsert_articles([art(1, "changed")])
    assert probe.unindexed_articles() == before + 1


def test_the_probe_counts_chunks_written_without_a_hash(store: Neo4jStore) -> None:
    from graphrag.adapters.health import Neo4jProbe

    probe = Neo4jProbe(URI, USER, _password())
    store.upsert_articles([art(1)])
    before = probe.unindexed_articles()
    store.upsert_chunks([indexed(1, 0, "some text", 0)])
    assert probe.unindexed_articles() == before  # chunks alone do not make it current
