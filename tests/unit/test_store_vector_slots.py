"""How the Neo4j store picks, records and cleans the vector slots of embedding models.

A fake driver answers each query from a small script, so no database is needed: the real
behaviour against Neo4j is covered by tests/integration/test_embedding_versions.py.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Self

import neo4j
import pytest

from graphrag.adapters import neo4j_store as store_module
from graphrag.adapters.neo4j_store import Neo4jStore
from graphrag.domain.models import Chunk, IndexedChunk, IndexInfo
from graphrag.domain.versioning import LEGACY_SLOT, VectorSlot, VersionNotReadyError

LEGACY = IndexInfo("legacy-model", "r1", 4, "title. chunk")
NEW = IndexInfo("new-model", "r2", 8, "title. chunk")
OTHER = IndexInfo("other-model", "r3", 6, "title. chunk")

Handler = Callable[[str, dict[str, Any]], list[dict[str, Any]]]


class Date:
    def to_native(self) -> datetime:
        return datetime(2026, 11, 20, tzinfo=UTC)


class Result:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.records)

    def single(self, strict: bool = False) -> dict[str, Any] | None:
        if not self.records and strict:
            raise AssertionError("no record")
        return self.records[0] if self.records else None

    def consume(self) -> None:
        return None


class Session:
    def __init__(self, driver: Driver) -> None:
        self.driver = driver

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def run(self, query: str, **params: Any) -> Result:
        self.driver.calls.append((query, params))
        return Result(self.driver.handler(query, params))

    def execute_write(self, work: Callable[..., Any], *args: Any) -> Any:
        return work(self, *args)


class Driver:
    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def session(self) -> Session:
        return Session(self)

    def close(self) -> None:
        self.closed = True

    def ran(self, needle: str) -> list[dict[str, Any]]:
        return [params for query, params in self.calls if needle in query]


def info_node(info: IndexInfo) -> dict[str, Any]:
    return {
        "model": info.model,
        "revision": info.revision,
        "dimensions": info.dimensions,
        "text_template": info.text_template,
    }


def fingerprint(info: IndexInfo | None, slot: VectorSlot = LEGACY_SLOT) -> list[dict[str, Any]]:
    """The answer to ACTIVE_FINGERPRINT: no row without an active version."""
    if info is None:
        return []
    row = {**info_node(info), "vector_property": slot.property, "vector_index": slot.index}
    return [row]


def open_store(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> tuple[Neo4jStore, Driver]:
    driver = Driver(handler)
    monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda *_a, **_k: driver)
    return Neo4jStore("bolt://x", "u", "p"), driver


def answers(
    *,
    recorded: IndexInfo | None,
    active: IndexInfo | None = None,
    active_slot: VectorSlot | None = None,
) -> Handler:
    """Answer the questions about the recorded model and the active version."""

    def handle(query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if query == store_module.READ_INDEX_INFO:
            return [] if recorded is None else [{"i": info_node(recorded)}]
        if query == store_module.ACTIVE_FINGERPRINT:
            return fingerprint(active, active_slot or LEGACY_SLOT)
        if query == store_module.VECTOR_INDEX_OPTIONS:
            return [{"options": {"indexConfig": {"vector.dimensions": 8}}}]
        return []

    return handle


def test_the_model_that_built_the_database_keeps_the_legacy_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, driver = open_store(monkeypatch, answers(recorded=LEGACY))
    store.upsert_chunks([], {"a": "h"}, LEGACY)
    assert store._slot_for(LEGACY) == LEGACY_SLOT
    assert store._slot_for(None) == LEGACY_SLOT  # no active version: the legacy slot
    assert driver.ran("indexed_hash")  # the write went through


def test_any_other_model_gets_a_slot_of_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    store, _driver = open_store(monkeypatch, answers(recorded=LEGACY))
    assert store._slot_for(NEW) == NEW.slot()


def test_without_a_recorded_model_the_first_one_takes_the_legacy_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _driver = open_store(monkeypatch, answers(recorded=None))
    assert store._slot_for(NEW) == LEGACY_SLOT


def test_the_active_slot_is_the_one_the_active_version_was_built_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _driver = open_store(
        monkeypatch, answers(recorded=LEGACY, active=NEW, active_slot=NEW.slot())
    )
    assert store._slot_for(None) == NEW.slot()


def test_the_active_model_is_the_version_s_or_else_the_recorded_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _driver = open_store(monkeypatch, answers(recorded=LEGACY, active=NEW))
    assert store.active_index_info() == NEW
    legacy_store, _ = open_store(monkeypatch, answers(recorded=LEGACY))
    assert legacy_store.active_index_info() == LEGACY
    unfingerprinted = answers(recorded=LEGACY)

    def version_without_model(query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if query == store_module.ACTIVE_FINGERPRINT:  # a version made before models existed
            return [{"model": None, "vector_property": "embedding", "vector_index": "chunk"}]
        return unfingerprinted(query, params)

    old_store, _ = open_store(monkeypatch, version_without_model)
    assert old_store.active_index_info() == LEGACY


def test_searches_are_checked_against_the_active_model_not_the_recorded_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _driver = open_store(monkeypatch, answers(recorded=LEGACY, active=NEW))
    store.check_index_info(NEW)
    with pytest.raises(ValueError, match="index --reembed"):
        store.check_index_info(LEGACY)
    nothing, _ = open_store(monkeypatch, answers(recorded=None))
    with pytest.raises(ValueError, match="no index fingerprint"):
        nothing.check_index_info(NEW)


def test_the_vector_index_of_the_new_model_is_created_with_its_own_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, driver = open_store(monkeypatch, answers(recorded=LEGACY))
    store.ensure_chunk_schema(8, NEW)
    created = [q for q, _ in driver.calls if q.lstrip().startswith("CREATE VECTOR INDEX")]
    assert created == [store_module.create_vector_index(8, NEW.slot())]
    assert NEW.slot().index in created[0]
    assert [p["name"] for p in driver.ran("SHOW VECTOR INDEXES")] == [NEW.slot().index]


def test_chunks_missing_a_vector_are_asked_for_in_the_slot_of_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handle(query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if query == store_module.READ_INDEX_INFO:
            return [{"i": info_node(LEGACY)}]
        return [{"key": "a#0#x", "text": "A. one"}, {"key": "b#0#y", "text": "B. two"}]

    store, driver = open_store(monkeypatch, handle)
    assert store.chunks_missing_vector(NEW, 2) == [("a#0#x", "A. one"), ("b#0#y", "B. two")]
    (params,) = driver.ran(f"c.{NEW.slot().property} IS NULL")
    assert params == {"limit": 2}


def test_vectors_are_written_only_where_missing_and_nothing_is_sent_for_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, driver = open_store(monkeypatch, answers(recorded=LEGACY))
    store.set_chunk_vectors(NEW, {})
    assert driver.ran("SET c.") == []
    store.set_chunk_vectors(NEW, {"a#0#x": (0.5, 1.0)})
    (params,) = driver.ran(f"SET c.{NEW.slot().property} = row.embedding")
    assert params == {"rows": [{"key": "a#0#x", "embedding": [0.5, 1.0]}]}
    assert f"WHERE c.{NEW.slot().property} IS NULL" in driver.calls[-1][0]


def test_chunks_are_written_into_the_slot_of_the_model_without_replacing_a_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, driver = open_store(monkeypatch, answers(recorded=LEGACY))
    chunk = IndexedChunk(Chunk("2401.00001", 0, "text"), "T. text", (1.0, 2.0))
    store.upsert_chunks([chunk], {"2401.00001": "h"}, NEW)
    (query,) = [q for q, _ in driver.calls if "MERGE (c:Chunk" in q]
    slot = NEW.slot().property
    assert f"SET c.{slot} = coalesce(c.{slot}, row.embedding)" in query
    assert "c.embedding =" not in query  # the legacy property is not touched


def test_a_search_reads_the_index_of_the_active_model(monkeypatch: pytest.MonkeyPatch) -> None:
    store, driver = open_store(
        monkeypatch, answers(recorded=LEGACY, active=NEW, active_slot=NEW.slot())
    )
    assert store.vector_search([0.0] * 8, 3) == []
    (params,) = driver.ran("db.index.vector.queryNodes")
    assert params["index"] == NEW.slot().index


def publication(
    *, recorded: IndexInfo | None, active: IndexInfo | None, covered: bool = True
) -> Handler:
    base = answers(recorded=recorded, active=active)
    fingerprint_seen: dict[str, Any] = {}

    def handle(query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if query == store_module.CREATE_VERSION:
            fingerprint_seen.update(params["fingerprint"])
            return [{"id": 7}]
        if query == store_module.COUNT_MANIFEST:
            return [
                {
                    "id": 7,
                    "created_at": Date(),
                    "chunk_count": 3,
                    "status": "pending",
                    "model": fingerprint_seen.get("model"),
                }
            ]
        if query == store_module.INDEX_STATES:
            return [
                {"name": params["vector_index"], "state": "ONLINE"},
                {"name": store_module.FULLTEXT_INDEX, "state": "ONLINE"},
            ]
        if "count(c) AS chunks" in query:
            return [{"chunks": 3, "with_vector": 3 if covered else 1, "with_text": 3}]
        return base(query, params)

    return handle


def test_publishing_with_a_model_records_its_fingerprint_and_checks_its_own_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, driver = open_store(monkeypatch, publication(recorded=LEGACY, active=LEGACY))
    version = store.publish_version(NEW)
    assert version.id == 7
    assert version.embedding_model == "new-model"
    (params,) = driver.ran("SET version += $fingerprint")
    assert params["fingerprint"] == {
        **info_node(NEW),
        "vector_property": NEW.slot().property,
        "vector_index": NEW.slot().index,
    }
    awaited = [p["name"] for p in driver.ran("db.awaitIndex")]
    assert awaited == [NEW.slot().index, store_module.FULLTEXT_INDEX]
    assert any(f"c.{NEW.slot().property} IS NOT NULL" in q for q, _ in driver.calls)  # coverage


def test_publishing_without_a_model_goes_on_with_the_active_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, driver = open_store(monkeypatch, publication(recorded=LEGACY, active=LEGACY))
    store.publish_version()
    (params,) = driver.ran("SET version += $fingerprint")
    assert params["fingerprint"]["vector_index"] == LEGACY_SLOT.index
    assert params["fingerprint"]["model"] == "legacy-model"


def test_publishing_with_nothing_recorded_makes_a_version_without_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, driver = open_store(monkeypatch, publication(recorded=None, active=None))
    store.publish_version()
    (params,) = driver.ran("SET version += $fingerprint")
    assert params["fingerprint"] == {}


def test_a_new_model_whose_vectors_are_missing_is_refused_before_any_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, driver = open_store(
        monkeypatch, publication(recorded=LEGACY, active=LEGACY, covered=False)
    )
    with pytest.raises(VersionNotReadyError, match="not in the vector index"):
        store.publish_version(NEW)
    assert driver.ran("SET version.status = 'active'") == []  # the pointer never moved


def test_a_missing_vector_index_of_the_new_model_is_named_in_the_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = publication(recorded=LEGACY, active=LEGACY)

    def handle(query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if query == store_module.INDEX_STATES:
            return [{"name": store_module.FULLTEXT_INDEX, "state": "ONLINE"}]
        return base(query, params)

    store, _driver = open_store(monkeypatch, handle)
    with pytest.raises(VersionNotReadyError, match=f"index {NEW.slot().index} does not exist"):
        store.publish_version(NEW)


def retention(
    *, kept_slots: list[VectorSlot], dropped_slots: list[VectorSlot], chunks: int = 4
) -> Handler:
    def version(version_id: int) -> dict[str, Any]:
        return {
            "id": version_id,
            "created_at": Date(),
            "chunk_count": 1,
            "status": "indexed",
            "model": None,
        }

    def rows(slots: list[VectorSlot]) -> list[dict[str, Any]]:
        return [{"property": s.property, "index": s.index} for s in slots]

    def handle(query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if query == store_module.CHAIN_VERSIONS:
            return [version(3), version(2), version(1)]
        if query == store_module._all_versions():
            return [version(3), version(2), version(1)]
        if query == store_module.VERSION_SLOTS:
            return rows(dropped_slots if params["ids"] == [1] else kept_slots)
        if "count(c) AS n" in query and "REMOVE" not in query and "DELETE" not in query:
            return [{"n": chunks}]
        if "DETACH DELETE c" in query or "REMOVE c." in query:
            return [{"n": 0}]
        return []

    return handle


def test_a_dry_run_names_the_slots_only_dropped_versions_used_and_deletes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = retention(kept_slots=[NEW.slot()], dropped_slots=[LEGACY_SLOT, NEW.slot()])
    store, driver = open_store(monkeypatch, handler)
    report = store.collect_garbage(keep=2, min_age_days=0, apply=False)
    assert report.dropped_vector_indexes == (LEGACY_SLOT.index,)
    assert report.dropped_versions == (1,)
    assert report.applied is False
    assert driver.ran("DROP INDEX") == []
    assert not any("DETACH DELETE" in q for q, _ in driver.calls)


def test_applying_drops_each_unused_index_and_strips_its_vectors_in_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strips = iter([1, 0, 0])  # the legacy slot needs two batches, the other one needs one

    def handle(query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if "REMOVE c." in query:
            return [{"n": next(strips)}]
        return base(query, params)

    base = retention(kept_slots=[NEW.slot()], dropped_slots=[LEGACY_SLOT, OTHER.slot()])
    store, driver = open_store(monkeypatch, handle)
    report = store.collect_garbage(keep=2, min_age_days=0, apply=True)
    assert report.dropped_vector_indexes == tuple(sorted((LEGACY_SLOT.index, OTHER.slot().index)))
    dropped = [q for q, _ in driver.calls if q.startswith("DROP INDEX")]
    assert sorted(dropped) == sorted(
        f"DROP INDEX {s.index} IF EXISTS" for s in (LEGACY_SLOT, OTHER.slot())
    )
    removed = [q for q, _ in driver.calls if "REMOVE c." in q]
    assert len(removed) == 3
    assert [q.count("REMOVE c.embedding\n") for q in removed] == [1, 1, 0]  # legacy first
    assert all(p["batch"] == store_module.GC_BATCH for q, p in driver.calls if "REMOVE c." in q)


def test_a_slot_a_kept_version_still_uses_is_never_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = retention(kept_slots=[LEGACY_SLOT], dropped_slots=[LEGACY_SLOT])
    store, driver = open_store(monkeypatch, handler)
    report = store.collect_garbage(keep=2, min_age_days=0, apply=True)
    assert report.dropped_vector_indexes == ()
    assert driver.ran("DROP INDEX") == []
