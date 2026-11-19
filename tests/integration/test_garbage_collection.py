"""Garbage collection of old versions and unreferenced chunks on the real Neo4j.

The sweep is restricted to the test articles (ids 9999.*) and to the versions made by the test,
so a run never touches real data.
"""

import neo4j
import pytest

from graphrag.adapters.neo4j_store import Neo4jStore
from tests.integration.test_chunk_store import _max_version_id, indexed, store, unit_vector
from tests.integration.test_neo4j_store import URI, USER, _password, _stored, art

pytestmark = pytest.mark.integration

__all__ = ["store"]  # the shared fixture, re-exported for pytest

ARTICLE = "9999.00001"


_ABOVE = [0]  # the highest version id that existed before the test


@pytest.fixture
def scoped(store: Neo4jStore, monkeypatch: pytest.MonkeyPatch) -> Neo4jStore:
    _ABOVE[0] = _max_version_id()
    monkeypatch.setattr(
        "graphrag.adapters.neo4j_store.GC_CHUNK_SCOPE", "AND c.arxiv_id STARTS WITH '9999.'"
    )
    monkeypatch.setattr("graphrag.adapters.neo4j_store.GC_VERSION_SCOPE", f"AND v.id > {_ABOVE[0]}")
    store.upsert_articles([art(1)])
    return store


def publish_text(store: Neo4jStore, text: str) -> str:
    """Give the article one chunk with this text, publish a version, return the chunk key."""
    chunk = indexed(1, 0, text, 0)
    store.upsert_chunks([chunk], {ARTICLE: f"hash of {text}"})
    store.publish_version()
    return chunk.key


def stored_keys() -> set[str]:
    return {r["k"] for r in _stored("MATCH (c:Chunk {arxiv_id: '9999.00001'}) RETURN c.key AS k")}


def stored_versions() -> int:
    return len(_stored("MATCH (v:Version) WHERE v.id > $above RETURN v.id", above=_ABOVE[0]))


def four_versions(store: Neo4jStore) -> list[str]:
    return [publish_text(store, f"text {n}") for n in range(1, 5)]


def test_a_dry_run_reports_what_would_go_and_deletes_nothing(scoped: Neo4jStore) -> None:
    four_versions(scoped)
    report = scoped.collect_garbage(keep=2, min_age_days=0, apply=False)
    assert report.applied is False
    assert len(report.kept_versions) == 2
    assert len(report.dropped_versions) == 2
    assert report.unreferenced_chunks == 2
    assert len(stored_keys()) == 4
    assert stored_versions() == 4


def test_applying_drops_old_versions_and_the_chunks_only_they_referenced(
    scoped: Neo4jStore,
) -> None:
    keys = four_versions(scoped)
    report = scoped.collect_garbage(keep=2, min_age_days=0, apply=True)
    assert report.applied is True
    assert (len(report.dropped_versions), report.unreferenced_chunks) == (2, 2)
    assert stored_keys() == set(keys[2:])
    assert stored_versions() == 2


def test_a_chunk_shared_with_a_kept_version_survives(scoped: Neo4jStore) -> None:
    other = indexed(1, 1, "stays", 1)
    scoped.upsert_chunks([indexed(1, 0, "old", 0), other], {ARTICLE: "h1"})
    scoped.publish_version()
    scoped.upsert_chunks([indexed(1, 0, "new", 0), other], {ARTICLE: "h2"})
    scoped.publish_version()
    scoped.upsert_chunks([indexed(1, 0, "newer", 0), other], {ARTICLE: "h3"})
    scoped.publish_version()
    scoped.collect_garbage(keep=2, min_age_days=0, apply=True)
    assert indexed(1, 0, "old", 0).key not in stored_keys()
    assert {other.key, indexed(1, 0, "new", 0).key, indexed(1, 0, "newer", 0).key} <= stored_keys()


def test_a_second_collection_has_nothing_left_to_do(scoped: Neo4jStore) -> None:
    four_versions(scoped)
    scoped.collect_garbage(keep=2, min_age_days=0, apply=True)
    again = scoped.collect_garbage(keep=2, min_age_days=0, apply=True)
    assert (again.dropped_versions, again.unreferenced_chunks) == ((), 0)


def test_chunks_written_by_a_run_that_has_not_published_yet_are_safe(scoped: Neo4jStore) -> None:
    four_versions(scoped)
    pending = indexed(1, 0, "pending", 0)
    scoped.upsert_chunks([pending], {ARTICLE: "pending hash"})  # no publish_version
    scoped.collect_garbage(keep=2, min_age_days=0, apply=True)
    assert pending.key in stored_keys()


def test_a_legacy_chunk_keyed_by_article_and_position_is_collected(scoped: Neo4jStore) -> None:
    four_versions(scoped)
    with neo4j.GraphDatabase.driver(URI, auth=(USER, _password())) as driver:
        driver.execute_query(
            "MATCH (a:Article {arxiv_id: $id}) "
            "CREATE (:Chunk {key: $key, arxiv_id: $id, position: 0, text: 't', "
            "search_text: 't'})-[:PART_OF]->(a)",
            id=ARTICLE,
            key=f"{ARTICLE}#0",
        )
    report = scoped.collect_garbage(keep=2, min_age_days=0, apply=True)
    assert f"{ARTICLE}#0" not in stored_keys()
    assert report.unreferenced_chunks == 3


def test_a_rollback_to_a_kept_version_still_works_after_collection(scoped: Neo4jStore) -> None:
    keys = four_versions(scoped)
    scoped.collect_garbage(keep=2, min_age_days=0, apply=True)
    rolled_back = scoped.rollback()
    assert rolled_back.chunk_count == 1
    hits = scoped.vector_search(unit_vector(0), 5)  # exact match, not drowned by real chunks
    assert [hit.text for hit in hits] == ["text 3"]
    assert keys[2] in stored_keys()


def test_nothing_older_than_the_chain_can_be_rolled_back_to_after_collection(
    scoped: Neo4jStore,
) -> None:
    four_versions(scoped)
    scoped.collect_garbage(keep=2, min_age_days=0, apply=True)
    scoped.rollback()
    with pytest.raises(LookupError):
        scoped.rollback()


def test_a_version_younger_than_the_minimum_age_is_kept_even_outside_the_window(
    scoped: Neo4jStore,
) -> None:
    keys = four_versions(scoped)
    report = scoped.collect_garbage(keep=2, min_age_days=1, apply=True)
    assert report.dropped_versions == ()
    assert stored_keys() == set(keys)


def test_a_version_abandoned_by_a_rollback_is_dropped_with_its_own_chunks(
    scoped: Neo4jStore,
) -> None:
    first = publish_text(scoped, "first")
    abandoned = publish_text(scoped, "abandoned")
    scoped.rollback()
    scoped.upsert_chunks([indexed(1, 0, "third", 0)], {ARTICLE: "third hash"})
    scoped.publish_version()
    report = scoped.collect_garbage(keep=2, min_age_days=0, apply=True)
    assert len(report.dropped_versions) == 1
    assert abandoned not in stored_keys()
    assert first in stored_keys()


def test_nothing_is_collected_before_any_version_was_published(scoped: Neo4jStore) -> None:
    with pytest.raises(LookupError, match="no version"):
        scoped.collect_garbage(keep=2, min_age_days=0, apply=True)


def test_keeping_fewer_than_two_versions_is_refused_before_deleting_anything(
    scoped: Neo4jStore,
) -> None:
    four_versions(scoped)
    with pytest.raises(ValueError, match="at least 2"):
        scoped.collect_garbage(keep=1, min_age_days=0, apply=True)
    assert len(stored_keys()) == 4


def test_the_sweep_runs_in_several_batches(
    scoped: Neo4jStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("graphrag.adapters.neo4j_store.GC_BATCH", 1)
    four_versions(scoped)
    report = scoped.collect_garbage(keep=2, min_age_days=0, apply=True)
    assert report.unreferenced_chunks == 2
    assert len(stored_keys()) == 2
