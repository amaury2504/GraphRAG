from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    @property
    def dimensions(self) -> int:
        """Length of every vector returned; the vector index must be created with it."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text, in the same order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Return the vector of a search question (some models treat queries differently)."""
        ...
