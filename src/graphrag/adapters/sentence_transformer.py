"""Local embeddings with sentence-transformers (BAAI/bge-small-en-v1.5 by default)."""

from collections.abc import Sequence
from typing import Any, Protocol

from graphrag.domain.chunking import TEXT_TEMPLATE
from graphrag.domain.models import IndexInfo

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Pinned model revision (Hugging Face commit), like the other dependencies.
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
# From the model card: prefix search questions only, never the passages.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Encoder(Protocol):
    """The part of sentence_transformers.SentenceTransformer this adapter uses."""

    def get_embedding_dimension(self) -> int | None: ...

    def encode_query(
        self,
        inputs: list[str],
        *,
        prompt: str | None,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> Any: ...

    def encode_document(
        self,
        inputs: list[str],
        *,
        prompt: str | None,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> Any: ...


class SentenceTransformerEmbedder:
    """Embedder port over a sentence-transformers model; vectors are unit length (cosine)."""

    def __init__(
        self,
        model: Encoder,
        *,
        query_instruction: str = QUERY_INSTRUCTION,
        batch_size: int = 32,
    ) -> None:
        dimensions = model.get_embedding_dimension()
        if dimensions is None:
            raise ValueError("the model does not report its embedding dimension")
        self._model = model
        self._dimensions = dimensions
        self._query_instruction = query_instruction
        self._batch_size = batch_size

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode_document(
            list(texts),
            prompt="",  # passages take no instruction
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return self._checked(vectors.tolist(), len(texts))

    def embed_query(self, text: str) -> list[float]:
        vectors = self._model.encode_query(
            [text],
            prompt=self._query_instruction,
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return self._checked(vectors.tolist(), 1)[0]

    def _checked(self, vectors: list[list[float]], expected: int) -> list[list[float]]:
        if len(vectors) != expected or any(len(v) != self._dimensions for v in vectors):
            raise ValueError(f"expected {expected} vectors of {self._dimensions} dimensions")
        return vectors


def default_index_info(dimensions: int) -> IndexInfo:
    """The fingerprint of an index built with the pinned model and the title+chunk text."""
    return IndexInfo(MODEL_NAME, MODEL_REVISION, dimensions, TEXT_TEMPLATE)


def load_default_embedder(*, cache_folder: str | None = None) -> SentenceTransformerEmbedder:
    """Load the pinned bge-small model (downloaded once, then read from the local cache)."""
    from sentence_transformers import SentenceTransformer  # heavy import, only when needed

    model = SentenceTransformer(
        MODEL_NAME, revision=MODEL_REVISION, device="cpu", cache_folder=cache_folder
    )
    return SentenceTransformerEmbedder(model)
