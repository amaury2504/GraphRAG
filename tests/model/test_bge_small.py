"""Runs the real bge-small model: downloads it once (~130 MB). Run with `pytest -m model`."""

import math

import pytest

from graphrag.adapters.sentence_transformer import load_default_embedder

pytestmark = pytest.mark.model


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_bge_small_vectors_rank_the_relevant_passage_first() -> None:
    embedder = load_default_embedder()
    assert embedder.dimensions == 384
    passages = [
        "Retrieval-augmented generation grounds a language model in retrieved documents.",
        "The recipe needs two eggs, flour and a pinch of salt.",
    ]
    docs = embedder.embed_documents(passages)
    query = embedder.embed_query("How can a language model use external documents?")
    for vector in [*docs, query]:
        assert len(vector) == 384
        assert math.isclose(math.sqrt(dot(vector, vector)), 1.0, rel_tol=1e-4)
    assert dot(query, docs[0]) > dot(query, docs[1])
