from collections.abc import Mapping, Sequence
from typing import Protocol

from graphrag.domain.models import ArticleIndexState, IndexedChunk, IndexInfo
from graphrag.domain.versioning import VersionInfo


class ChunkStore(Protocol):
    def ensure_chunk_schema(self, dimensions: int) -> None:
        """Create the chunk key constraint, the vector index and the full-text index.

        Fails if a vector index already exists with another number of dimensions.
        """
        ...

    def ensure_index_info(self, info: IndexInfo) -> None:
        """Record what the vectors are built with, or fail if another model/template is recorded."""
        ...

    def articles_with_index_state(self) -> list[ArticleIndexState]:
        """Every article with the hash it was last indexed from, ordered by arXiv id."""
        ...

    def upsert_chunks(
        self, chunks: Sequence[IndexedChunk], article_hashes: Mapping[str, str] | None = None
    ) -> None:
        """Write the chunks and their PART_OF relations in one transaction. Idempotent.

        Chunks are immutable: one whose key already exists is left as it is, and none is ever
        deleted here (unreferenced ones are the garbage collector's business). With
        `article_hashes` (arXiv id -> hash the chunks were built from), each listed article
        records, in the same transaction, which of `chunks` are its own and the hash it was
        indexed from. An article with no chunk in `chunks` is thereby emptied.

        Written chunks are not readable until `publish_version` puts them in a version.
        """
        ...

    def publish_version(self) -> VersionInfo:
        """Snapshot the chunks of every indexed article as a new version and make it current.

        The new manifest and the move of the "current version" pointer happen in one
        transaction: a reader sees the old version or the new one, never a mix.
        """
        ...

    def current_version(self) -> VersionInfo | None:
        """The version searches read from, or None if none was ever published."""
        ...

    def rollback(self) -> VersionInfo:
        """Make the version published before the current one current again, and return it.

        Raises LookupError if there is no current version or no earlier one. Articles whose
        chunks differ from that version are marked as not indexed, so the next indexing run
        rebuilds them from their source text instead of silently bringing them back.
        """
        ...

    def chunk_count(self) -> int: ...
