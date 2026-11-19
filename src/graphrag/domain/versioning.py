"""Identity of a chunk and description of an index version.

A chunk is immutable: its key contains a digest of what it holds, so a changed chunk is a new
chunk with a new key, and an unchanged one keeps its key and is shared by every version that
contains it. A version is a snapshot (a manifest of chunk keys) of the whole corpus.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime

KEY_DIGEST_LENGTH = 16  # hex characters of the SHA-256 kept in the key (64 bits)


def chunk_key(arxiv_id: str, position: int, search_text: str) -> str:
    """`arxiv_id#position#digest`, the digest covering the article, the position and the text."""
    payload = "\0".join((arxiv_id, str(position), search_text)).encode("utf-8")
    return f"{arxiv_id}#{position}#{hashlib.sha256(payload).hexdigest()[:KEY_DIGEST_LENGTH]}"


@dataclass(frozen=True, slots=True)
class VersionInfo:
    """A published snapshot of the index: its number, when it was made, how many chunks it holds."""

    id: int
    created_at: datetime
    chunk_count: int
