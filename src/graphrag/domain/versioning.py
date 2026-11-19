"""Identity of a chunk and description of an index version.

A chunk is immutable: its key contains a digest of what it holds, so a changed chunk is a new
chunk with a new key, and an unchanged one keeps its key and is shared by every version that
contains it. A version is a snapshot (a manifest of chunk keys) of the whole corpus.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

KEY_DIGEST_LENGTH = 16  # hex characters of the SHA-256 kept in the key (64 bits)


def chunk_key(arxiv_id: str, position: int, search_text: str) -> str:
    """`arxiv_id#position#digest`, the digest covering the article, the position and the text."""
    payload = "\0".join((arxiv_id, str(position), search_text)).encode("utf-8")
    return f"{arxiv_id}#{position}#{hashlib.sha256(payload).hexdigest()[:KEY_DIGEST_LENGTH]}"


class VersionStatus(StrEnum):
    """Where a version is on its way to being read.

    `pending`: the manifest is being built. `indexed`: every chunk of the manifest is in the
    vector and the full-text index. `active`: the version searches read from (only one).
    """

    PENDING = "pending"
    INDEXED = "indexed"
    ACTIVE = "active"


@dataclass(frozen=True, slots=True)
class VersionInfo:
    """A snapshot of the index: its number, when it was made, how many chunks it holds.

    The status defaults to `active`, the one a published or current version has.
    """

    id: int
    created_at: datetime
    chunk_count: int
    status: VersionStatus = VersionStatus.ACTIVE


class VersionNotReadyError(RuntimeError):
    """A version cannot be made readable: part of its chunks is missing from an index."""

    def __init__(self, version_id: int, reason: str) -> None:
        super().__init__(f"version {version_id} is not ready to be read: {reason}")
        self.version_id = version_id


def ensure_indexed(version_id: int, *, chunks: int, with_vector: int, with_text: int) -> None:
    """Raise unless all `chunks` of the version are in the vector index and the full-text index."""
    problems = [
        f"{chunks - found} of {chunks} chunks are not in the {index} index"
        for index, found in (("vector", with_vector), ("full-text", with_text))
        if found != chunks
    ]
    if problems:
        raise VersionNotReadyError(version_id, "; ".join(problems))
