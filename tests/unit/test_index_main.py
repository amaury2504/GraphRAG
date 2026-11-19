from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import ClassVar, Self

import pytest

import graphrag.cli.index as index_cli
from graphrag.domain.models import IndexInfo
from graphrag.domain.versioning import VersionInfo

VERSION = VersionInfo(3, datetime(2026, 11, 18, tzinfo=UTC), 12)


class RecordingStore:
    """Stands in for Neo4jStore: remembers how it was opened and what was asked of it."""

    opened: ClassVar[list[tuple[str, str, str]]] = []
    rollback_error: ClassVar[LookupError | None] = None

    def __init__(self, uri: str, user: str, password: str) -> None:
        RecordingStore.opened.append((uri, user, password))
        self.rolled_back = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        pass

    def rollback(self) -> VersionInfo:
        if RecordingStore.rollback_error is not None:
            raise RecordingStore.rollback_error
        self.rolled_back = True
        return VERSION


class StubEmbedder:
    dimensions = 2


@pytest.fixture(autouse=True)
def stubbed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    RecordingStore.opened = []
    RecordingStore.rollback_error = None
    secret = SimpleNamespace(get_secret_value=lambda: "secret")
    settings = SimpleNamespace(neo4j_uri="bolt://db", neo4j_user="neo", neo4j_password=secret)
    monkeypatch.setattr(index_cli, "Settings", lambda: settings)
    monkeypatch.setattr(index_cli, "Neo4jStore", RecordingStore)
    monkeypatch.setattr(index_cli, "log_line", lambda _message: None)


def test_rollback_opens_the_store_and_rolls_back_without_loading_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden() -> None:
        raise AssertionError("rolling back must not load the embedding model")

    monkeypatch.setattr(index_cli, "load_default_embedder", forbidden)
    assert index_cli.main(["--rollback"]) == 0
    assert RecordingStore.opened == [("bolt://db", "neo", "secret")]


def test_rollback_with_nothing_to_go_back_to_exits_with_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    RecordingStore.rollback_error = LookupError("no earlier version to roll back to")
    lines: list[str] = []
    monkeypatch.setattr(index_cli, "log_line", lines.append)
    assert index_cli.main(["--rollback"]) == 2
    assert lines == ["chunks: no earlier version to roll back to"]


def test_indexing_runs_with_the_embedder_and_prints_the_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: dict[str, object] = {}

    def fake_run(store: object, embedder: object, **kwargs: object) -> index_cli.IndexReport:
        calls.update(store=store, embedder=embedder, **kwargs)
        return index_cli.IndexReport(articles=1, chunks=2, total_chunks=3, version=4)

    info = IndexInfo("m", "r", 2, "title. chunk")
    monkeypatch.setattr(index_cli, "load_default_embedder", StubEmbedder)
    monkeypatch.setattr(index_cli, "default_index_info", lambda _dimensions: info)
    monkeypatch.setattr(index_cli, "run_indexing", fake_run)
    assert index_cli.main(["--limit", "5", "--batch-size", "7"]) == 0
    assert (calls["limit"], calls["batch_size"], calls["info"]) == (5, 7, info)
    assert isinstance(calls["store"], RecordingStore)
    assert capsys.readouterr().out.strip() == (
        "IndexReport(articles=1, chunks=2, total_chunks=3, version=4)"
    )
