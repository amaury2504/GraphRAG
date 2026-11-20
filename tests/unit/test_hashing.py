from datetime import date

import pytest

from graphrag.domain.chunking import (
    CHUNKER_VERSION,
    DEFAULT_MAX_WORDS,
    ChunkerConfig,
    chunk_abstract,
)
from graphrag.domain.hashing import article_hash, chunker_fingerprint, content_hash
from graphrag.domain.models import Article


def article(title: str = "Title", abstract: str = "Abstract.") -> Article:
    return Article("2401.00001", title, abstract, date(2025, 1, 1), ("cs.CL",), ("Ann",))


def test_hash_is_a_stable_sha256_hex_digest() -> None:
    first = content_hash("Title", "Abstract.")
    assert first == content_hash("Title", "Abstract.")
    assert len(first) == 64
    assert int(first, 16) >= 0


def test_hash_changes_when_the_title_changes() -> None:
    assert content_hash("A", "x") != content_hash("B", "x")


def test_hash_changes_when_the_abstract_changes() -> None:
    assert content_hash("A", "x") != content_hash("A", "y")


def test_hash_does_not_confuse_where_the_title_ends() -> None:
    assert content_hash("ab", "c") != content_hash("a", "bc")


def test_hash_ignores_whitespace_differences() -> None:
    assert content_hash("A  title", "Some\n text.") == content_hash(" A title ", "Some text.")


def test_hash_ignores_what_the_chunks_do_not_depend_on() -> None:
    other = Article("2401.00001", "Title", "Abstract.", date(2020, 5, 5), ("cs.IR",), ("Bob",))
    assert article_hash(article()) == article_hash(other)


def test_article_hash_is_the_content_hash_of_its_title_and_abstract() -> None:
    assert article_hash(article("T", "A")) == content_hash("T", "A")


def test_the_chunker_fingerprint_names_its_version_and_size() -> None:
    assert chunker_fingerprint() == f"{CHUNKER_VERSION}/{DEFAULT_MAX_WORDS}"


def test_a_new_chunker_version_changes_every_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    before = content_hash("T", "A")
    monkeypatch.setattr("graphrag.domain.hashing.CHUNKER_VERSION", "next")
    assert content_hash("T", "A") != before


def test_the_chunker_output_is_pinned_so_a_change_forces_a_version_bump() -> None:
    """If this fails, the chunking changed: raise CHUNKER_VERSION, then update this test."""
    abstract = "First sentence here. Second one follows. " + "Third " * 5 + "ends now."
    chunks = chunk_abstract("2401.00001", abstract, max_words=8)
    assert [c.text for c in chunks] == [
        "First sentence here. Second one follows.",
        "Third Third Third Third Third ends now.",
    ]
    assert CHUNKER_VERSION == "1"


def test_default_config_keeps_the_historical_fingerprint() -> None:
    assert chunker_fingerprint(ChunkerConfig()) == chunker_fingerprint()
    assert chunker_fingerprint() == f"{CHUNKER_VERSION}/{DEFAULT_MAX_WORDS}"


def test_a_different_size_or_overlap_changes_the_fingerprint_and_the_hash() -> None:
    base = chunker_fingerprint()
    assert chunker_fingerprint(ChunkerConfig(max_words=100)) != base
    assert chunker_fingerprint(ChunkerConfig(overlap_sentences=0)) != base
    assert chunker_fingerprint(ChunkerConfig(overlap_sentences=2)) != chunker_fingerprint(
        ChunkerConfig(overlap_sentences=0)
    )
    assert content_hash("T", "A", ChunkerConfig(max_words=100)) != content_hash("T", "A")
    assert article_hash(article(), ChunkerConfig(max_words=100)) == content_hash(
        "Title", "Abstract.", ChunkerConfig(max_words=100)
    )
