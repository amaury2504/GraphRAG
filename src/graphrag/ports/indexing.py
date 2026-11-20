from collections.abc import Mapping, Sequence
from typing import Protocol

from graphrag.domain.models import ArticleIndexState, IndexedChunk, IndexInfo
from graphrag.domain.versioning import GcReport, VersionInfo


class ChunkStore(Protocol):
    def ensure_chunk_schema(self, dimensions: int, info: IndexInfo | None = None) -> None:
        """Create the chunk key constraint, the vector index and the full-text index.

        The vector index is the one of the model `info` (of the active model when None): each
        model has its own, built beside the others. Fails if it already exists with another
        number of dimensions.
        """
        ...

    def ensure_index_info(self, info: IndexInfo) -> None:
        """Record what the vectors are built with, or fail if the active vectors differ."""
        ...

    def active_index_info(self) -> IndexInfo | None:
        """What the vectors searches read were built with (None if nothing is recorded)."""
        ...

    def chunks_missing_vector(self, info: IndexInfo, limit: int) -> list[tuple[str, str]]:
        """Up to `limit` (key, search text) of the chunks the next version holds that have no
        vector of the model `info` yet."""
        ...

    def set_chunk_vectors(self, info: IndexInfo, vectors: Mapping[str, Sequence[float]]) -> None:
        """Add the vectors of the model `info` to chunks that have none (never replaces one)."""
        ...

    def articles_with_index_state(self) -> list[ArticleIndexState]:
        """Every article with the hash it was last indexed from, ordered by arXiv id."""
        ...

    def upsert_chunks(
        self,
        chunks: Sequence[IndexedChunk],
        article_hashes: Mapping[str, str] | None = None,
        info: IndexInfo | None = None,
    ) -> None:
        """Write the chunks and their PART_OF relations in one transaction. Idempotent.

        Chunks are immutable: one whose key already exists is left as it is, and none is ever
        deleted here (unreferenced ones are the garbage collector's business). With
        `article_hashes` (arXiv id -> hash the chunks were built from), each listed article
        records, in the same transaction, which of `chunks` are its own and the hash it was
        indexed from. An article with no chunk in `chunks` is thereby emptied.

        The vectors are those of the model `info` (the active one when None); a chunk that exists
        without a vector of that model gains it, one that has it keeps it.

        Written chunks are not readable until `publish_version` puts them in a version.
        """
        ...

    def publish_version(self, info: IndexInfo | None = None) -> VersionInfo:
        """Snapshot the chunks of every indexed article as a new version and make it current.

        The version is `pending` while its manifest is built, `indexed` once every chunk of the
        manifest is in the vector and the full-text index, and `active` once the "current
        version" pointer moves to it (the version it replaces becomes `indexed` again). The
        pointer move is one transaction: a reader sees the old version or the new one, never a
        mix. The version is read with the vectors of the model `info` (the active one when None),
        so publishing with a new model switches to it and a rollback switches back. Raises
        VersionNotReadyError, leaving the active version as it was, if an index is not online
        or does not hold every chunk.
        """
        ...

    def current_version(self) -> VersionInfo | None:
        """The version searches read from, or None if none was ever published."""
        ...

    def rollback(self) -> VersionInfo:
        """Make the version published before the current one current again, and return it.

        Only an `indexed` version can be rolled back to; the one abandoned becomes `indexed`.
        Raises LookupError if there is no current version or no earlier one. Articles whose
        chunks differ from that version are marked as not indexed, so the next indexing run
        rebuilds them from their source text instead of silently bringing them back.
        """
        ...

    def collect_garbage(self, *, keep: int, min_age_days: int, apply: bool) -> GcReport:
        """Mark and sweep: drop old versions, then the chunks that nothing references any more.

        Roots are the manifests of the versions kept (the `keep` newest of the current chain, and
        any other version younger than `min_age_days`) and the chunks every article owns for the
        next version, so chunks written by a run that has not published yet are safe. Without
        `apply` nothing is deleted and the report says what would be.

        Raises LookupError if no version was ever published (nothing says what is live) and
        ValueError if `keep` or `min_age_days` is out of range. Idempotent and resumable.
        """
        ...

    def chunk_count(self) -> int: ...
