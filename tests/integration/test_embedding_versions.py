"""Embedding model versioning on the real Neo4j: a second model is built next to the first.

Uses only test articles (ids 9999.*) and the test version pointer. The second and third models
are fake (8 and 6 dimensions) so their derived vector indexes are dropped afterwards; the real
legacy index (`chunk_embedding`) is never garbage collected here.
"""

from collections.abc import Iterator

import neo4j
import pytest

from graphrag.adapters.neo4j_store import Neo4jStore
from graphrag.domain.models import Chunk, IndexedChunk, IndexInfo
from graphrag.domain.versioning import VersionNotReadyError
from tests.integration.test_chunk_store import (
    DIMENSIONS,
    _max_version_id,
    indexed,
    store,
    unit_vector,
)
from tests.integration.test_neo4j_store import URI, USER, _password, _stored, art

pytestmark = pytest.mark.integration

__all__ = ["store"]  # the shared fixture, re-exported for pytest

B = IndexInfo("test-model-b", "rev1", 8, "title. chunk")
C = IndexInfo("test-model-c", "rev1", 6, "title. chunk")


def vector_b(hot: int) -> list[float]:
    return [1.0 if i == hot else 0.0 for i in range(8)]


def vector_c(hot: int) -> list[float]:
    return [1.0 if i == hot else 0.0 for i in range(6)]


def indexed_b(n: int, position: int, text: str, hot: int) -> IndexedChunk:
    chunk = Chunk(f"9999.{n:05d}", position, text)
    return IndexedChunk(chunk, f"T{n}. {text}", tuple(vector_b(hot)))


def vector_indexes() -> set[str]:
    return {r["name"] for r in _stored("SHOW VECTOR INDEXES YIELD name RETURN name")}


@pytest.fixture
def models(store: Neo4jStore, monkeypatch: pytest.MonkeyPatch) -> Iterator[Neo4jStore]:
    # versions, re-embedding and rollbacks only look at the test articles, never at real ones
    monkeypatch.setattr(
        "graphrag.adapters.neo4j_store.ARTICLE_SCOPE", "AND a.arxiv_id STARTS WITH '9999.'"
    )
    # A database with no recorded model (an empty one, as in CI) gets a legacy model for the
    # test, removed afterwards: the first model recorded owns the legacy slot.
    recorded_here = store.read_index_info() is None
    if recorded_here:
        store.ensure_index_info(IndexInfo("test-legacy-model", "rev0", DIMENSIONS, "title. chunk"))
    yield store
    with neo4j.GraphDatabase.driver(URI, auth=(USER, _password())) as driver:
        for info in (B, C):
            driver.execute_query(f"DROP INDEX {info.slot().index} IF EXISTS")
        if recorded_here:
            driver.execute_query("MATCH (i:IndexInfo {name: 'chunks'}) DELETE i")


def published_with_legacy(store: Neo4jStore) -> None:
    store.upsert_articles([art(1), art(2)])
    store.upsert_chunks(
        [indexed(1, 0, "alpha", 0), indexed(2, 0, "beta", 1)],
        {"9999.00001": "h1", "9999.00002": "h2"},
    )
    store.publish_version()


def embed_everything(store: Neo4jStore, info: IndexInfo, vectors: dict[str, list[float]]) -> None:
    store.ensure_chunk_schema(info.dimensions, info)
    missing = store.chunks_missing_vector(info, 100)
    store.set_chunk_vectors(info, {key: vectors[text] for key, text in missing if text in vectors})


def test_a_second_model_gets_its_own_index_with_its_own_dimensions(models: Neo4jStore) -> None:
    models.ensure_chunk_schema(8, B)
    models.ensure_chunk_schema(8, B)  # repeatable
    assert {B.slot().index, "chunk_embedding"} <= vector_indexes()
    with pytest.raises(ValueError, match=f"{B.slot().index} has 8 dimensions, the embedder has 5"):
        models.ensure_chunk_schema(5, B)


def test_the_active_model_is_the_recorded_one_until_a_version_says_otherwise(
    models: Neo4jStore,
) -> None:
    recorded = models.read_index_info()
    assert models.active_index_info() == recorded
    published_with_legacy(models)
    assert models.active_index_info() == recorded  # the version adopts the recorded fingerprint
    assert models.current_version() is not None


def test_chunks_missing_a_vector_are_those_of_the_articles_without_it(models: Neo4jStore) -> None:
    published_with_legacy(models)
    models.ensure_chunk_schema(8, B)
    missing = models.chunks_missing_vector(B, 100)
    ours = sorted(text for _key, text in missing if text in ("T1. alpha", "T2. beta"))
    assert ours == ["T1. alpha", "T2. beta"]
    assert len(models.chunks_missing_vector(B, 1)) == 1  # the limit is honoured


def test_the_new_model_is_built_beside_the_old_one_then_switched_atomically(
    models: Neo4jStore,
) -> None:
    published_with_legacy(models)
    first = models.current_version()
    assert first is not None
    embed_everything(models, B, {"T1. alpha": vector_b(2), "T2. beta": vector_b(3)})
    # building in parallel: the readers still use the old model and its index
    assert models.active_index_info() == models.read_index_info()
    assert [h.text for h in models.vector_search(unit_vector(0), 3)][:1] == ["alpha"]
    second = models.publish_version(B)
    assert second.id == first.id + 1
    assert second.embedding_model == "test-model-b"
    assert models.active_index_info() == B
    assert [h.text for h in models.vector_search(vector_b(3), 3)][:1] == ["beta"]


def test_a_new_model_with_missing_vectors_is_refused_and_nothing_changes(
    models: Neo4jStore,
) -> None:
    published_with_legacy(models)
    before = models.current_version()
    models.ensure_chunk_schema(8, B)
    with pytest.raises(VersionNotReadyError, match="not in the vector index"):
        models.publish_version(B)
    assert models.current_version() == before
    assert models.active_index_info() == models.read_index_info()


def test_a_rollback_gives_back_the_previous_model(models: Neo4jStore) -> None:
    published_with_legacy(models)
    recorded = models.read_index_info()
    assert recorded is not None
    embed_everything(models, B, {"T1. alpha": vector_b(2), "T2. beta": vector_b(3)})
    models.publish_version(B)
    back = models.rollback()
    assert back.embedding_model == recorded.model
    assert models.active_index_info() == recorded
    assert [h.text for h in models.vector_search(unit_vector(1), 3)][:1] == ["beta"]


def test_the_searches_are_checked_against_the_active_model(models: Neo4jStore) -> None:
    published_with_legacy(models)
    recorded = models.read_index_info()
    assert recorded is not None
    embed_everything(models, B, {"T1. alpha": vector_b(2), "T2. beta": vector_b(3)})
    models.publish_version(B)
    models.check_index_info(B)
    with pytest.raises(ValueError, match="built differently"):
        models.check_index_info(recorded)
    models.ensure_index_info(B)  # indexing with the active model is accepted


def test_new_chunks_of_the_new_model_carry_only_its_vector(models: Neo4jStore) -> None:
    models.ensure_chunk_schema(8, B)
    models.upsert_articles([art(1)])
    models.upsert_chunks([indexed_b(1, 0, "gamma", 4)], {"9999.00001": "h1"}, B)
    rows = _stored(
        "MATCH (c:Chunk {arxiv_id: '9999.00001'}) "
        f"RETURN c.embedding AS old, size(c.{B.slot().property}) AS new"
    )
    assert [(r["old"], r["new"]) for r in rows] == [(None, 8)]


def test_rewriting_an_existing_chunk_adds_the_missing_vector_and_keeps_the_old_one(
    models: Neo4jStore,
) -> None:
    models.ensure_chunk_schema(8, B)
    models.upsert_articles([art(1)])
    models.upsert_chunks([indexed(1, 0, "delta", 0)], {"9999.00001": "h1"})
    models.upsert_chunks([indexed_b(1, 0, "delta", 5)], {"9999.00001": "h1"}, B)
    rows = _stored(
        "MATCH (c:Chunk {arxiv_id: '9999.00001'}) "
        f"RETURN size(c.embedding) AS old, c.embedding[0] AS first, "
        f"size(c.{B.slot().property}) AS new"
    )
    assert [(r["old"], r["first"], r["new"]) for r in rows] == [(DIMENSIONS, 1.0, 8)]


def publish_three(store: Neo4jStore) -> None:
    """Version 1 is built with model B, versions 2 and 3 with model C."""
    store.ensure_chunk_schema(8, B)
    store.upsert_articles([art(1)])
    store.upsert_chunks([indexed_b(1, 0, "epsilon", 0)], {"9999.00001": "h1"}, B)
    store.publish_version(B)
    store.ensure_chunk_schema(6, C)
    store.set_chunk_vectors(C, {k: vector_c(0) for k, _ in store.chunks_missing_vector(C, 100)})
    store.publish_version(C)
    store.publish_version(C)


@pytest.fixture
def scoped(models: Neo4jStore, monkeypatch: pytest.MonkeyPatch) -> Neo4jStore:
    above = _max_version_id()
    monkeypatch.setattr(
        "graphrag.adapters.neo4j_store.GC_CHUNK_SCOPE", "AND c.arxiv_id STARTS WITH '9999.'"
    )
    monkeypatch.setattr("graphrag.adapters.neo4j_store.GC_VERSION_SCOPE", f"AND v.id > {above}")
    return models


def test_a_dry_run_names_the_vector_index_nobody_needs_any_more(scoped: Neo4jStore) -> None:
    publish_three(scoped)
    report = scoped.collect_garbage(keep=2, min_age_days=0, apply=False)
    assert report.dropped_vector_indexes == (B.slot().index,)
    assert B.slot().index in vector_indexes()


def test_applying_drops_the_unused_index_and_its_vectors_but_keeps_the_used_one(
    scoped: Neo4jStore,
) -> None:
    publish_three(scoped)
    report = scoped.collect_garbage(keep=2, min_age_days=0, apply=True)
    assert report.dropped_vector_indexes == (B.slot().index,)
    assert B.slot().index not in vector_indexes()
    assert C.slot().index in vector_indexes()
    rows = _stored(
        "MATCH (c:Chunk {arxiv_id: '9999.00001'}) "
        f"RETURN c.{B.slot().property} AS b, size(c.{C.slot().property}) AS c"
    )
    assert [(r["b"], r["c"]) for r in rows] == [(None, 6)]


def test_an_index_no_version_has_used_yet_is_left_alone(scoped: Neo4jStore) -> None:
    publish_three(scoped)
    other = IndexInfo("test-model-b", "rev9", 8, "title. chunk")
    scoped.ensure_chunk_schema(8, other)  # a re-embedding in progress: no version uses it
    try:
        report = scoped.collect_garbage(keep=2, min_age_days=0, apply=True)
        assert other.slot().index not in report.dropped_vector_indexes
        assert other.slot().index in vector_indexes()
    finally:
        with neo4j.GraphDatabase.driver(URI, auth=(USER, _password())) as driver:
            driver.execute_query(f"DROP INDEX {other.slot().index} IF EXISTS")
