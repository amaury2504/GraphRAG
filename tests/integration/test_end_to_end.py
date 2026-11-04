"""Question -> FastAPI -> hybrid search on the real Neo4j -> fake LLM -> cited answer.

Only the language model and the embedder are fakes (no network, deterministic). Test articles
use ids 9999.* and are removed afterwards; they live in the real indexes next to the corpus,
so the assertions only concern them.
"""

import hashlib
import math
import re
from collections.abc import Iterator, Sequence
from datetime import date

import pytest
from fastapi.testclient import TestClient

from graphrag.adapters.neo4j_store import Neo4jStore
from graphrag.api.app import create_app
from graphrag.cli.index import index_articles
from graphrag.domain.models import Article, Completion, Usage
from graphrag.pipeline.answer import QuestionAnswerer
from graphrag.pipeline.retrieval import HybridRetriever, VectorRetriever
from tests.integration.test_neo4j_store import URI, USER, _cleanup, _password

pytestmark = pytest.mark.integration

DIMENSIONS = 384
FIRST_PART = (
    "Quarkfilter uses a learned gate to skip needless lookups. " + "Filler sentence here. " * 70
)
SECOND_PART = "Results show a tenfold speedup on the zebrabench suite."


class HashingEmbedder:
    """Bag of words hashed into 384 dimensions: same text, same vector; no model download."""

    dimensions = DIMENSIONS

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMENSIONS] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class CitingLLM:
    """Cites the numbered source whose text mentions 'zebrabench', as a careful model would."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, system: str, user: str) -> Completion:
        self.prompts.append(user)
        blocks = re.split(r"\n\n(?=\[\d+\] )", user.split("Sources:\n\n", 1)[1])
        for block in blocks:
            if "zebrabench" in block:
                return Completion(f"It skips lookups [{block.split(']', 1)[0][1:]}].", Usage(10, 5))
        return Completion("The sources do not say.", Usage(10, 5))


ARTICLES = [
    Article(
        "9999.90001",
        "Learned gates",
        FIRST_PART + SECOND_PART,
        date(2025, 5, 1),
        ("cs.IR",),
        (),
    ),
    Article(
        "9999.90002",
        "Unrelated gardening notes",
        "Tomatoes need sun and water. " * 20,
        date(2025, 5, 2),
        ("cs.CL",),
        (),
    ),
]


@pytest.fixture
def store() -> Iterator[Neo4jStore]:
    _cleanup()
    with Neo4jStore(URI, USER, _password()) as s:
        s.ensure_schema()
        s.ensure_chunk_schema(DIMENSIONS)
        s.upsert_articles(ARTICLES)
        s.upsert_chunks(index_articles(ARTICLES, HashingEmbedder()))
        yield s
    _cleanup()


def client(store: Neo4jStore, llm: CitingLLM) -> TestClient:
    embedder = HashingEmbedder()
    answerer = QuestionAnswerer(
        {"vector": VectorRetriever(store, embedder), "hybrid": HybridRetriever(store, embedder)},
        llm,
    )
    return TestClient(create_app([], lambda: answerer))


def test_a_question_gets_an_answer_citing_the_expected_article(store: Neo4jStore) -> None:
    llm = CitingLLM()
    response = client(store, llm).post(
        "/ask", json={"question": "How does Quarkfilter skip lookups?", "configuration": "hybrid"}
    )
    assert response.status_code == 200
    body = response.json()
    assert "[arXiv:9999.90001]" in body["answer"]
    assert [c["arxiv_id"] for c in body["citations"]] == ["9999.90001"]
    assert "9999.90001" in [r["arxiv_id"] for r in body["retrieved"]]
    assert body["timings"]["total_seconds"] >= 0
    assert body["usage"]["total_tokens"] == 15


def test_the_model_reads_the_whole_abstract_not_only_the_matching_chunk(store: Neo4jStore) -> None:
    llm = CitingLLM()
    body = (
        client(store, llm)
        .post("/ask", json={"question": "Quarkfilter learned gate", "configuration": "hybrid"})
        .json()
    )
    ours = [r for r in body["retrieved"] if r["arxiv_id"] == "9999.90001"]
    assert ours
    assert "zebrabench" not in ours[0]["text"]  # the matching chunk lacks the word
    # 'zebrabench' sits in the second chunk; the question matches the first one
    assert "zebrabench" in llm.prompts[0]


def test_vector_configuration_answers_through_the_same_route(store: Neo4jStore) -> None:
    # The hashing embedder's vectors are not comparable with the corpus' real ones, so only
    # the route and the shape are checked here; ranking is covered by the store tests.
    response = client(store, CitingLLM()).post(
        "/ask", json={"question": "Quarkfilter learned gate", "configuration": "vector", "k": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["configuration"] == "vector"
    assert 1 <= len(body["retrieved"]) <= 3  # the CI database holds only the test articles
