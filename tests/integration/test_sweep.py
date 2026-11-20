"""The chunk sweep on the throwaway database, with a hashing embedder (no model download)."""

from collections.abc import Iterator

import pytest

from graphrag.adapters.neo4j_store import Neo4jStore
from graphrag.cli.sweep import sweep_corpus
from graphrag.domain.chunking import TEXT_TEMPLATE, ChunkerConfig
from graphrag.domain.evaluation import EvalQuestion
from graphrag.domain.models import IndexInfo
from graphrag.pipeline.sweep import BASELINE
from tests.integration.test_end_to_end import ARTICLES, DIMENSIONS, HashingEmbedder
from tests.integration.test_neo4j_store import URI, USER, _cleanup, _password

pytestmark = pytest.mark.integration

INFO = IndexInfo("hashing-test", "r1", DIMENSIONS, TEXT_TEMPLATE)
QUESTIONS = [
    EvalQuestion(
        "factual-01",
        "factual",
        "How does Quarkfilter skip lookups?",
        "a gate",
        ("9999.90001",),
        {},
        True,
    )
]


@pytest.fixture
def store() -> Iterator[Neo4jStore]:
    _cleanup()
    with Neo4jStore(URI, USER, _password()) as s:
        yield s
    _cleanup()


def test_each_configuration_is_indexed_in_turn_and_evaluated(store: Neo4jStore) -> None:
    logs: list[str] = []
    points = sweep_corpus(
        ARTICLES,
        store,
        HashingEmbedder(),
        INFO,
        QUESTIONS,
        [BASELINE, ChunkerConfig(max_words=40, overlap_sentences=0)],
        (1, 5),
        log=logs.append,
    )
    assert [p.config for p in points] == [BASELINE, ChunkerConfig(40, 0)]
    assert points[1].chunks > points[0].chunks  # smaller chunks: more of them
    assert all(set(p.results) == {"vector", "hybrid"} for p in points)
    assert all(len(r) == 1 for p in points for r in p.results.values())
    version = store.current_version()
    assert version is not None
    assert (version.id, version.chunk_count) == (2, points[1].chunks)
    assert any("w40/o0" in line for line in logs)
