from datetime import UTC, datetime

import pytest

from graphrag.domain.models import Chunk, IndexedChunk
from graphrag.domain.versioning import VersionInfo, chunk_key


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
