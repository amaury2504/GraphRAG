from datetime import UTC, datetime

import pytest

from graphrag.domain.models import Chunk, IndexedChunk
from graphrag.domain.versioning import (
    VersionInfo,
    VersionNotReadyError,
    VersionStatus,
    chunk_key,
    ensure_indexed,
)


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


def test_a_version_carries_one_of_three_statuses_and_is_active_by_default() -> None:
    created = datetime(2026, 11, 18, tzinfo=UTC)
    assert VersionInfo(1, created, 0).status is VersionStatus.ACTIVE
    assert [s.value for s in VersionStatus] == ["pending", "indexed", "active"]
    assert VersionInfo(1, created, 0, VersionStatus.PENDING).status is VersionStatus.PENDING


def test_a_version_not_ready_names_the_version_and_the_missing_part() -> None:
    error = VersionNotReadyError(4, "3 of 5 chunks have no vector")
    assert error.version_id == 4
    assert str(error) == "version 4 is not ready to be read: 3 of 5 chunks have no vector"


def test_a_version_whose_chunks_are_all_in_both_indexes_is_ready() -> None:
    ensure_indexed(4, chunks=5, with_vector=5, with_text=5)
    ensure_indexed(5, chunks=0, with_vector=0, with_text=0)  # an empty corpus is a valid version


@pytest.mark.parametrize(
    ("with_vector", "with_text", "message"),
    [
        (3, 5, "2 of 5 chunks are not in the vector index"),
        (5, 1, "4 of 5 chunks are not in the full-text index"),
        (
            0,
            0,
            "5 of 5 chunks are not in the vector index; "
            "5 of 5 chunks are not in the full-text index",
        ),
    ],
)
def test_a_version_with_chunks_missing_from_an_index_is_refused(
    with_vector: int, with_text: int, message: str
) -> None:
    with pytest.raises(VersionNotReadyError) as raised:
        ensure_indexed(4, chunks=5, with_vector=with_vector, with_text=with_text)
    assert str(raised.value) == f"version 4 is not ready to be read: {message}"
