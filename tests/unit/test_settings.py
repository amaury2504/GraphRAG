import pytest
from pydantic import ValidationError

from graphrag.settings import Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "POSTGRES_DSN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir("/")  # avoid picking up a local .env file


def test_loads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://neo4j:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://u:p@db/x")
    s = Settings()  # type: ignore[call-arg]
    assert s.neo4j_uri == "bolt://neo4j:7687"
    assert s.neo4j_user == "neo4j"
    assert s.neo4j_password.get_secret_value() == "pw"


def test_missing_secrets_fail_fast() -> None:
    with pytest.raises(ValidationError) as exc:
        Settings()  # type: ignore[call-arg]
    missing = {e["loc"][0] for e in exc.value.errors()}
    assert missing == {"neo4j_password", "postgres_dsn"}


def test_secrets_not_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_PASSWORD", "topsecret")
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://u:dbsecret@db/x")
    text = repr(Settings())  # type: ignore[call-arg]
    assert "topsecret" not in text
    assert "dbsecret" not in text
