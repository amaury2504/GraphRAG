from datetime import UTC, datetime, timedelta

import pytest

from graphrag.domain.models import Chunk, IndexedChunk
from graphrag.domain.versioning import VersionInfo, chunk_key, plan_retention


def test_the_key_is_deterministic() -> None:
    assert chunk_key("2401.00001", 0, "T. text") == chunk_key("2401.00001", 0, "T. text")


def test_the_key_starts_with_the_article_and_the_position() -> None:
    assert chunk_key("2401.00001", 3, "T. text").startswith("2401.00001#3#")


@pytest.mark.parametrize(
    ("arxiv_id", "position", "text"),
    [("2401.00002", 0, "T. text"), ("2401.00001", 1, "T. text"), ("2401.00001", 0, "T. other")],
)
def test_the_key_changes_with_any_of_its_parts(arxiv_id: str, position: int, text: str) -> None:
    assert chunk_key(arxiv_id, position, text) != chunk_key("2401.00001", 0, "T. text")


def test_an_indexed_chunk_is_keyed_by_its_content_not_only_its_position() -> None:
    old = IndexedChunk(Chunk("2401.00001", 0, "old"), "T. old", (1.0,))
    new = IndexedChunk(Chunk("2401.00001", 0, "new"), "T. new", (1.0,))
    same = IndexedChunk(Chunk("2401.00001", 0, "old"), "T. old", (2.0,))
    assert old.key != new.key  # a changed chunk is a new immutable chunk
    assert old.key == same.key  # the embedding does not enter the key (T5 versions it)
    assert old.key == chunk_key("2401.00001", 0, "T. old")


def test_a_version_describes_itself() -> None:
    created = datetime(2026, 11, 18, tzinfo=UTC)
    info = VersionInfo(id=2, created_at=created, chunk_count=10)
    assert (info.id, info.created_at, info.chunk_count) == (2, created, 10)


NOW = datetime(2026, 11, 19, tzinfo=UTC)


def version(version_id: int, age_days: int) -> VersionInfo:
    return VersionInfo(version_id, NOW - timedelta(days=age_days), 10)


CHAIN = [version(5, 1), version(4, 10), version(3, 20), version(2, 30), version(1, 40)]


def test_the_newest_versions_of_the_chain_are_kept_and_the_older_ones_dropped() -> None:
    plan = plan_retention(CHAIN, [], keep=3, min_age_days=0, now=NOW)
    assert plan.kept == (5, 4, 3)
    assert plan.dropped == (2, 1)


def test_a_chain_shorter_than_the_window_drops_nothing() -> None:
    plan = plan_retention(CHAIN[:2], [], keep=3, min_age_days=0, now=NOW)
    assert plan.kept == (5, 4)
    assert plan.dropped == ()


def test_keeping_fewer_than_two_versions_is_refused_because_a_rollback_needs_the_previous() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        plan_retention(CHAIN, [], keep=1, min_age_days=0, now=NOW)


def test_a_version_outside_the_window_but_younger_than_the_minimum_age_is_kept() -> None:
    plan = plan_retention(CHAIN, [], keep=2, min_age_days=25, now=NOW)
    assert plan.kept == (5, 4, 3)  # version 3 is 20 days old, version 2 is 30
    assert plan.dropped == (2, 1)


def test_versions_abandoned_by_a_rollback_are_dropped_unless_recent() -> None:
    abandoned = [version(7, 2), version(6, 50)]
    plan = plan_retention(CHAIN[:3], abandoned, keep=2, min_age_days=5, now=NOW)
    assert plan.kept == (5, 4, 7)
    assert plan.dropped == (3, 6)


def test_a_negative_minimum_age_is_refused() -> None:
    with pytest.raises(ValueError, match="minimum age"):
        plan_retention(CHAIN, [], keep=3, min_age_days=-1, now=NOW)


def test_a_version_exactly_as_old_as_the_minimum_age_is_dropped() -> None:
    plan = plan_retention(CHAIN[:3], [], keep=2, min_age_days=20, now=NOW)
    assert plan.dropped == (3,)
