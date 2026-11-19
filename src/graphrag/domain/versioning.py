"""Identity of a chunk and description of an index version.

A chunk is immutable: its key contains a digest of what it holds, so a changed chunk is a new
chunk with a new key, and an unchanged one keeps its key and is shared by every version that
contains it. A version is a snapshot (a manifest of chunk keys) of the whole corpus.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

KEY_DIGEST_LENGTH = 16  # hex characters of the SHA-256 kept in the key (64 bits)
MIN_KEPT_VERSIONS = 2  # the current one and the one a rollback goes back to


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


@dataclass(frozen=True, slots=True)
class RetentionPlan:
    """Which versions the garbage collector keeps and which it drops, as version ids."""

    kept: tuple[int, ...]
    dropped: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GcReport:
    """What a garbage collection removed (or would remove, when `applied` is False)."""

    kept_versions: tuple[int, ...]
    dropped_versions: tuple[int, ...]
    unreferenced_chunks: int
    applied: bool


def plan_retention(
    chain: Sequence[VersionInfo],
    abandoned: Sequence[VersionInfo],
    *,
    keep: int,
    min_age_days: int,
    now: datetime,
) -> RetentionPlan:
    """Keep the `keep` newest versions of the chain, plus any other version younger than the limit.

    `chain` runs from the current version back through its predecessors, newest first.
    `abandoned` holds the versions off that chain (those a rollback left behind). A version
    outside the window is dropped once it is at least `min_age_days` old.
    """
    if keep < MIN_KEPT_VERSIONS:
        raise ValueError(f"keep at least {MIN_KEPT_VERSIONS} versions, a rollback needs one")
    if min_age_days < 0:
        raise ValueError("the minimum age cannot be negative")
    limit = timedelta(days=min_age_days)
    kept = [v.id for v in chain[:keep]]
    dropped: list[int] = []
    for candidate in (*chain[keep:], *abandoned):
        (kept if now - candidate.created_at < limit else dropped).append(candidate.id)
    return RetentionPlan(tuple(kept), tuple(dropped))
