import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from graphrag.adapters.http import RequestFailedError
from graphrag.api.app import (
    LazyAnswerer,
    NotConfiguredError,
    build_answerer,
    create_app,
)
from graphrag.domain.models import Answer, Completion, SearchHit, Usage
from graphrag.pipeline.answer import QuestionAnswerer
from graphrag.settings import Settings

HITS = [
    SearchHit("2401.00001", "Title 1", 0, "chunk 1", 0.9),
    SearchHit("2401.00002", "Title 2", 1, "chunk 2", 0.8),
]


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, k: int) -> list[SearchHit]:
        self.calls.append((question, k))
        return HITS


class FakeLLM:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def complete(self, system: str, user: str) -> Completion:
        if self.error is not None:
            raise self.error
        return Completion("It works [2].", Usage(120, 30))


def client(*, error: Exception | None = None, retriever: FakeRetriever | None = None) -> TestClient:
    ticks = iter([10.0, 10.5, 12.5])  # start, retrieved, answered
    qa = QuestionAnswerer(
        {"vector": retriever or FakeRetriever(), "hybrid": retriever or FakeRetriever()},
        FakeLLM(error),
        clock=lambda: next(ticks),
    )
    return TestClient(create_app([], lambda: qa), raise_server_exceptions=False)


def test_ask_returns_answer_citations_sources_and_usage() -> None:
    response = client().post("/ask", json={"question": "What works?"})
    assert response.status_code == 200
    assert response.json() == {
        "question": "What works?",
        "configuration": "hybrid",
        "answer": "It works [arXiv:2401.00002].",
        "citations": [
            {"arxiv_id": "2401.00002", "title": "Title 2", "position": 1, "text": "chunk 2"}
        ],
        "retrieved": [
            {"arxiv_id": "2401.00001", "title": "Title 1", "position": 0, "text": "chunk 1"},
            {"arxiv_id": "2401.00002", "title": "Title 2", "position": 1, "text": "chunk 2"},
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
        "timings": {"retrieval_seconds": 0.5, "llm_seconds": 2.0, "total_seconds": 2.5},
    }


def test_mode_and_k_reach_the_retriever() -> None:
    retriever = FakeRetriever()
    response = client(retriever=retriever).post(
        "/ask", json={"question": "  q  ", "configuration": "vector", "k": 7}
    )
    assert response.status_code == 200
    assert response.json()["configuration"] == "vector"
    assert retriever.calls == [("q", 7)]  # the question is trimmed


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 2001},
        {"question": "q", "configuration": "graph"},
        {"question": "q", "k": 0},
        {"question": "q", "k": 11},
        {"question": "q", "k": "many"},
    ],
)
def test_invalid_requests_are_422(body: dict[str, object]) -> None:
    assert client().post("/ask", json=body).status_code == 422


def test_ask_is_post_only() -> None:
    assert client().get("/ask").status_code == 405


def test_ask_is_503_when_answering_is_not_wired() -> None:
    response = TestClient(create_app([])).post("/ask", json={"question": "q"})
    assert response.status_code == 503


def test_ask_is_503_when_the_key_is_missing() -> None:
    def missing() -> QuestionAnswerer:
        raise NotConfiguredError("MISTRAL_API_KEY is not set")

    response = TestClient(create_app([], missing)).post("/ask", json={"question": "q"})
    assert response.status_code == 503
    assert response.json() == {"detail": "MISTRAL_API_KEY is not set"}


@pytest.mark.parametrize("error", [RequestFailedError("url sk-secret"), ValueError("bad shape")])
def test_language_model_failure_is_502_without_details(error: Exception) -> None:
    response = client(error=error).post("/ask", json={"question": "q"})
    assert response.status_code == 502
    assert response.json() == {"detail": "the language model is unavailable"}
    assert "sk-secret" not in response.text


def test_health_still_works_next_to_ask() -> None:
    assert client().get("/health").status_code == 200


def test_lazy_answerer_builds_once() -> None:
    built: list[int] = []

    def build() -> QuestionAnswerer:
        built.append(1)
        return QuestionAnswerer({}, FakeLLM())

    lazy = LazyAnswerer(build)
    assert built == []
    assert lazy() is lazy()
    assert built == [1]


def test_lazy_answerer_retries_after_a_failed_build() -> None:
    attempts: list[int] = []

    def build() -> QuestionAnswerer:
        attempts.append(1)
        if len(attempts) == 1:
            raise NotConfiguredError("no key yet")
        return QuestionAnswerer({}, FakeLLM())

    lazy = LazyAnswerer(build)
    with pytest.raises(NotConfiguredError):
        lazy()
    lazy()
    assert len(attempts) == 2


@pytest.mark.parametrize("key", [None, SecretStr("")])
def test_build_answerer_needs_a_key(key: SecretStr | None) -> None:
    settings = Settings(
        neo4j_password=SecretStr("x"), postgres_dsn=SecretStr("x"), mistral_api_key=key
    )
    with pytest.raises(NotConfiguredError, match="MISTRAL_API_KEY"):
        build_answerer(settings)


def test_answer_is_a_plain_value() -> None:
    answer = Answer("q", "t", (), (), Usage())
    assert answer.usage.total_tokens == 0


def test_preload_builds_the_answerer_in_the_background_and_swallows_failures() -> None:
    import threading

    from graphrag.api.app import preload

    built = threading.Event()

    def build() -> QuestionAnswerer:
        built.set()
        raise NotConfiguredError("no key")

    preload(build)
    assert built.wait(5)
