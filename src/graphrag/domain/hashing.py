"""Fingerprint of what an article's chunks are built from, to tell which articles changed."""

import hashlib

from graphrag.domain.chunking import (
    CHUNKER_VERSION,
    DEFAULT_OVERLAP_SENTENCES,
    ChunkerConfig,
)
from graphrag.domain.models import Article


def chunker_fingerprint(config: ChunkerConfig | None = None) -> str:
    """The chunker version and its settings: any change to either rebuilds every chunk.

    The default overlap adds nothing, so the fingerprint of the default settings never changed.
    """
    config = config or ChunkerConfig()
    fingerprint = f"{CHUNKER_VERSION}/{config.max_words}"
    if config.overlap_sentences != DEFAULT_OVERLAP_SENTENCES:
        fingerprint += f"/overlap{config.overlap_sentences}"
    return fingerprint


def content_hash(title: str, abstract: str, config: ChunkerConfig | None = None) -> str:
    """SHA-256 of title + abstract (whitespace-normalised) + the chunker fingerprint."""
    parts = (chunker_fingerprint(config), " ".join(title.split()), " ".join(abstract.split()))
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def article_hash(article: Article, config: ChunkerConfig | None = None) -> str:
    return content_hash(article.title, article.abstract, config)
