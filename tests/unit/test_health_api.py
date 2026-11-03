import pytest
from fastapi.testclient import TestClient

from graphrag.api.app import create_app


class FakeProbe:
    def __init__(self, name: str, error: Exception | None = None) -> None:
        self.name = name
        self._error = error
        self.calls = 0

    def check(self) -> None:
        self.calls += 1
        if self._error is not None:
            raise self._error


def client_for(*probes: FakeProbe) -> TestClient:
    return TestClient(create_app(list(probes)))


def test_all_services_up_returns_200_ok() -> None:
    response = client_for(FakeProbe("neo4j"), FakeProbe("postgres")).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "services": {"neo4j": "ok", "postgres": "ok"}}


@pytest.mark.parametrize("down", ["neo4j", "postgres"])
def test_one_service_down_returns_503_degraded(down: str) -> None:
    probes = [
        FakeProbe(n, ConnectionError("boom") if n == down else None) for n in ("neo4j", "postgres")
    ]
    response = client_for(*probes).get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["services"][down] == "down"
    assert {v for k, v in body["services"].items() if k != down} == {"ok"}


def test_all_services_down() -> None:
    response = client_for(FakeProbe("neo4j", OSError()), FakeProbe("postgres", RuntimeError())).get(
        "/health"
    )
    assert response.status_code == 503
    assert response.json()["services"] == {"neo4j": "down", "postgres": "down"}


def test_error_details_are_not_leaked() -> None:
    secret = "password=hunter2 host=internal"
    response = client_for(FakeProbe("postgres", RuntimeError(secret))).get("/health")
    assert "hunter2" not in response.text
    assert "internal" not in response.text


def test_every_probe_is_checked_on_each_call() -> None:
    first, second = FakeProbe("a", RuntimeError()), FakeProbe("b")
    client = client_for(first, second)
    client.get("/health")
    client.get("/health")
    assert (first.calls, second.calls) == (2, 2)


def test_no_probes_reports_ok() -> None:
    response = TestClient(create_app([])).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "services": {}}


def test_unknown_route_is_404() -> None:
    assert client_for().get("/nope").status_code == 404
