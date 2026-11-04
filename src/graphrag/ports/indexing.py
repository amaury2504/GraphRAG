from collections.abc import Sequence
from typing import Protocol

from graphrag.domain.models import Article, IndexedChunk


class ChunkStore(Protocol):
    def ensure_chunk_schema(self, dimensions: int) -> None:
        """Create the chunk key constraint, the vector index and the full-text index.

        Fails if a vector index already exists with another number of dimensions.
        """
        ...

    def articles_without_chunks(self, limit: int | None = None) -> list[Article]:
        """Articles that have no chunk yet, ordered by arXiv id."""
        ...

    def upsert_chunks(self, chunks: Sequence[IndexedChunk]) -> None:
        """Write the chunks and their PART_OF relations in one transaction. Idempotent."""
        ...

    def chunk_count(self) -> int: ...
