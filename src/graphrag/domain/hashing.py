"""Fingerprint of what an article's chunks are built from, to tell which articles changed."""

import hashlib

from graphrag.domain.chunking import CHUNKER_VERSION, DEFAULT_MAX_WORDS
from graphrag.domain.models import Article


def chunker_fingerprint() -> str:
    """The chunker version and its size setting: any change to either rebuilds every chunk."""
    return f"{CHUNKER_VERSION}/{DEFAULT_MAX_WORDS}"


def content_hash(title: str, abstract: str) -> str:
    """SHA-256 of title + abstract (whitespace-normalised) + the chunker fingerprint."""
    parts = (chunker_fingerprint(), " ".join(title.split()), " ".join(abstract.split()))
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def article_hash(article: Article) -> str:
    return content_hash(article.title, article.abstract)
