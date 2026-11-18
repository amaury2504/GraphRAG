from collections.abc import Mapping, Sequence
from typing import Protocol

from graphrag.domain.models import ArticleIndexState, IndexedChunk, IndexInfo


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
    ) -> int:
        """Write the chunks and their PART_OF relations in one transaction. Idempotent.

        With `article_hashes` (arXiv id -> hash the chunks were built from), each listed
        article is replaced as a whole in the same transaction: its chunks that are not in
        `chunks` are deleted, and the hash is recorded as indexed. An article with no chunk
        in `chunks` is thereby emptied. Returns how many chunks were deleted.
        """
        ...

    def chunk_count(self) -> int: ...
