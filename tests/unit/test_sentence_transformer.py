from typing import Any

import pytest

from graphrag.adapters.sentence_transformer import (
    QUERY_INSTRUCTION,
    SentenceTransformerEmbedder,
)
from graphrag.ports.embedding import Embedder


class Vectors:
    """Mimics the numpy array returned by sentence-transformers (only tolist is used)."""

    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows

    def tolist(self) -> list[list[float]]:
        return self.rows


class FakeEncoder:
    def __init__(self, dimensions: int | None = 3, rows: int | None = None) -> None:
        self.dimensions = dimensions
        self.rows = rows  # force a wrong number of returned vectors
        self.calls: list[tuple[str, list[str], dict[str, Any]]] = []

    def get_embedding_dimension(self) -> int | None:
        return self.dimensions

    def _encode(self, kind: str, inputs: list[str], kwargs: dict[str, Any]) -> Vectors:
        self.calls.append((kind, inputs, kwargs))
        n = len(inputs) if self.rows is None else self.rows
        return Vectors([[float(i), 0.0, 1.0] for i in range(n)])

    def encode_query(self, inputs: list[str], **kwargs: Any) -> Vectors:
        return self._encode("query", inputs, kwargs)

    def encode_document(self, inputs: list[str], **kwargs: Any) -> Vectors:
        return self._encode("document", inputs, kwargs)


def test_satisfies_the_embedder_port() -> None:
    embedder: Embedder = SentenceTransformerEmbedder(FakeEncoder())
    assert embedder.dimensions == 3


def test_documents_are_encoded_without_instruction_and_normalized() -> None:
    encoder = FakeEncoder()
    vectors = SentenceTransformerEmbedder(encoder, batch_size=8).embed_documents(("a", "b"))
    assert vectors == [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    (kind, inputs, kwargs) = encoder.calls[0]
    assert (kind, inputs) == ("document", ["a", "b"])
    assert kwargs["prompt"] == ""
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["batch_size"] == 8


def test_query_gets_the_bge_instruction() -> None:
    encoder = FakeEncoder()
    vector = SentenceTransformerEmbedder(encoder).embed_query("what is RAG?")
    assert vector == [0.0, 0.0, 1.0]
    (kind, inputs, kwargs) = encoder.calls[0]
    assert (kind, inputs) == ("query", ["what is RAG?"])
    assert kwargs["prompt"] == "Represent this sentence for searching relevant passages: "
    assert kwargs["prompt"] == QUERY_INSTRUCTION
    assert kwargs["normalize_embeddings"] is True


def test_no_documents_means_no_model_call() -> None:
    encoder = FakeEncoder()
    assert SentenceTransformerEmbedder(encoder).embed_documents([]) == []
    assert encoder.calls == []


def test_unknown_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="dimension"):
        SentenceTransformerEmbedder(FakeEncoder(dimensions=None))


def test_wrong_vector_count_is_an_error() -> None:
    with pytest.raises(ValueError, match="expected 2 vectors"):
        SentenceTransformerEmbedder(FakeEncoder(rows=1)).embed_documents(["a", "b"])


def test_wrong_vector_length_is_an_error() -> None:
    with pytest.raises(ValueError, match="of 4 dimensions"):
        SentenceTransformerEmbedder(FakeEncoder(dimensions=4)).embed_query("q")
