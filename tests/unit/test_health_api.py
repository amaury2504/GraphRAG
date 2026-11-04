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
    assert response.json() == {
        "status": "ok",
        "services": {"neo4j": "ok", "postgres": "ok"},
        "unindexed_articles": None,
    }


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
    assert response.json() == {"status": "ok", "services": {}, "unindexed_articles": None}


def test_unknown_route_is_404() -> None:
    assert client_for().get("/nope").status_code == 404


def test_health_reports_unindexed_articles_when_neo4j_is_up() -> None:
    app = create_app([FakeProbe("neo4j")], unindexed=lambda: 7)
    body = TestClient(app).get("/health").json()
    assert body["unindexed_articles"] == 7
    assert body["status"] == "ok"


def test_unindexed_count_is_not_asked_when_neo4j_is_down() -> None:
    calls: list[int] = []

    def count() -> int:
        calls.append(1)
        return 1

    app = create_app([FakeProbe("neo4j", ConnectionError("x"))], unindexed=count)
    response = TestClient(app).get("/health")
    assert response.json()["unindexed_articles"] is None
    assert calls == []


def test_a_failing_unindexed_count_never_breaks_health() -> None:
    def broken() -> int:
        raise RuntimeError("boom")

    response = TestClient(create_app([FakeProbe("neo4j")], unindexed=broken)).get("/health")
    assert response.status_code == 200
    assert response.json()["unindexed_articles"] is None


def test_unindexed_is_none_when_not_wired() -> None:
    assert client_for(FakeProbe("neo4j")).get("/health").json()["unindexed_articles"] is None
