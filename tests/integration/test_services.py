"""Integration tests: need the compose services (`docker compose up -d --wait`).

Run with `pytest -m integration`. Connection settings come from the environment
(see `.env.example`).
"""

import os

import pytest
from fastapi.testclient import TestClient

from graphrag.adapters.health import Neo4jProbe, PostgresProbe
from graphrag.api.app import create_app

pytestmark = pytest.mark.integration


def _env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        pytest.fail(f"{key} must be set to run integration tests")
    return value


@pytest.fixture
def neo4j_probe() -> Neo4jProbe:
    return Neo4jProbe(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        os.environ.get("NEO4J_USER", "neo4j"),
        _env("NEO4J_PASSWORD"),
    )


@pytest.fixture
def postgres_probe() -> PostgresProbe:
    return PostgresProbe(_env("POSTGRES_DSN"))


def test_real_services_are_healthy(neo4j_probe: Neo4jProbe, postgres_probe: PostgresProbe) -> None:
    response = TestClient(create_app([neo4j_probe, postgres_probe])).get("/health")
    assert response.status_code == 200
    assert response.json()["services"] == {"neo4j": "ok", "postgres": "ok"}


def test_neo4j_wrong_password_is_down() -> None:
    probe = Neo4jProbe(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"), "neo4j", "definitely-wrong"
    )
    response = TestClient(create_app([probe])).get("/health")
    assert response.status_code == 503
    assert response.json()["services"] == {"neo4j": "down"}


def test_neo4j_unreachable_is_down() -> None:
    response = TestClient(create_app([Neo4jProbe("bolt://localhost:1", "neo4j", "x")])).get(
        "/health"
    )
    assert response.status_code == 503


def test_postgres_unreachable_is_down() -> None:
    probe = PostgresProbe("postgresql://u:p@localhost:1/db")
    response = TestClient(create_app([probe])).get("/health")
    assert response.status_code == 503
    assert response.json()["services"] == {"postgres": "down"}
