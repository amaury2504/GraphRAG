import logging
import threading
from collections.abc import Callable, Sequence
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field, StringConstraints

from graphrag.adapters.health import Neo4jProbe, PostgresProbe
from graphrag.adapters.http import RequestFailedError
from graphrag.domain.models import Answer, SearchHit
from graphrag.pipeline.answer import DEFAULT_K, QuestionAnswerer
from graphrag.ports.health import HealthProbe
from graphrag.settings import Settings

logger = logging.getLogger(__name__)


class NotConfiguredError(Exception):
    """A part of the application needs a setting that is missing."""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    services: dict[str, Literal["ok", "down"]]


class AskRequest(BaseModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]
    mode: Literal["vector", "hybrid"] = "hybrid"
    k: int = Field(default=DEFAULT_K, ge=1, le=10, description="number of articles retrieved")


class SourceOut(BaseModel):
    arxiv_id: str
    title: str
    position: int
    text: str

    @classmethod
    def of(cls, hit: SearchHit) -> SourceOut:
        return cls(arxiv_id=hit.arxiv_id, title=hit.title, position=hit.position, text=hit.text)


class UsageOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AskResponse(BaseModel):
    question: str
    mode: Literal["vector", "hybrid"]
    answer: str
    citations: list[SourceOut]
    retrieved: list[SourceOut]
    usage: UsageOut

    @classmethod
    def of(cls, answer: Answer, mode: Literal["vector", "hybrid"]) -> AskResponse:
        return cls(
            question=answer.question,
            mode=mode,
            answer=answer.text,
            citations=[SourceOut.of(h) for h in answer.citations],
            retrieved=[SourceOut.of(h) for h in answer.retrieved],
            usage=UsageOut(
                prompt_tokens=answer.usage.prompt_tokens,
                completion_tokens=answer.usage.completion_tokens,
                total_tokens=answer.usage.total_tokens,
            ),
        )


def create_app(
    probes: Sequence[HealthProbe],
    answerer: Callable[[], QuestionAnswerer] | None = None,
) -> FastAPI:
    """`answerer` is called on each /ask; it may build its parts lazily (model loading is slow)."""
    app = FastAPI(title="GraphRAG")

    @app.get("/health", response_model=HealthResponse)
    def health(response: Response) -> HealthResponse:
        services: dict[str, Literal["ok", "down"]] = {}
        for probe in probes:
            try:
                probe.check()
                services[probe.name] = "ok"
            except Exception:
                services[probe.name] = "down"
        healthy = all(state == "ok" for state in services.values())
        if not healthy:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="ok" if healthy else "degraded", services=services)

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        if answerer is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "question answering is off")
        try:
            question_answerer = answerer()
        except NotConfiguredError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        try:
            answer = question_answerer.ask(request.question, request.mode, request.k)
        except RequestFailedError, ValueError:
            logger.exception("the language model call failed")
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY, "the language model is unavailable"
            ) from None
        return AskResponse.of(answer, request.mode)

    return app


class LazyAnswerer:
    """Builds the answerer on the first /ask, so /health and startup stay fast."""

    def __init__(self, build: Callable[[], QuestionAnswerer]) -> None:
        self._build = build
        self._lock = threading.Lock()
        self._value: QuestionAnswerer | None = None

    def __call__(self) -> QuestionAnswerer:
        with self._lock:
            if self._value is None:
                self._value = self._build()
            return self._value


def build_answerer(settings: Settings) -> QuestionAnswerer:
    key = settings.mistral_api_key.get_secret_value() if settings.mistral_api_key else ""
    if not key:
        raise NotConfiguredError("MISTRAL_API_KEY is not set")
    # Imported here: they pull in torch, which /health does not need.
    from graphrag.adapters.mistral import MistralLLM, mistral_client
    from graphrag.adapters.neo4j_store import Neo4jStore
    from graphrag.adapters.sentence_transformer import load_default_embedder
    from graphrag.pipeline.retrieval import HybridRetriever, VectorRetriever
    from graphrag.pipeline.usage import MeteredLLM, UsageMeter

    store = Neo4jStore(
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password.get_secret_value()
    )
    embedder = load_default_embedder()
    llm = MeteredLLM(
        MistralLLM(mistral_client(key), model=settings.mistral_model),
        UsageMeter(settings.token_alert),
    )
    return QuestionAnswerer(
        {"vector": VectorRetriever(store, embedder), "hybrid": HybridRetriever(store, embedder)},
        llm,
    )


def create_default_app() -> FastAPI:
    settings = Settings()  # type: ignore[call-arg]  # values come from the environment
    return create_app(
        [
            Neo4jProbe(
                settings.neo4j_uri,
                settings.neo4j_user,
                settings.neo4j_password.get_secret_value(),
            ),
            PostgresProbe(settings.postgres_dsn.get_secret_value()),
        ],
        LazyAnswerer(lambda: build_answerer(settings)),
    )
