from collections.abc import Sequence
from typing import Literal

from fastapi import FastAPI, Response, status
from pydantic import BaseModel

from graphrag.adapters.health import Neo4jProbe, PostgresProbe
from graphrag.ports.health import HealthProbe
from graphrag.settings import Settings


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    services: dict[str, Literal["ok", "down"]]


def create_app(probes: Sequence[HealthProbe]) -> FastAPI:
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

    return app


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
        ]
    )
