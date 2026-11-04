from collections.abc import Sequence
from typing import Protocol

from graphrag.domain.models import SearchHit


class ChunkSearch(Protocol):
    """The two chunk indexes. Hits come best first; one article may appear several times."""

    def vector_search(self, vector: Sequence[float], k: int) -> list[SearchHit]:
        """The k chunks closest in meaning to the vector (cosine score)."""
        ...

    def keyword_search(self, text: str, k: int) -> list[SearchHit]:
        """The k chunks that best match the words of `text` (plain text, not query syntax)."""
        ...


class Retriever(Protocol):
    def retrieve(self, question: str, k: int) -> list[SearchHit]:
        """At most k articles, best first, each with its best chunk."""
        ...
