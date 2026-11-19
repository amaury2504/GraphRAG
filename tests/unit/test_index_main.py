from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import ClassVar, Self

import pytest

import graphrag.cli.index as index_cli
from graphrag.domain.models import IndexInfo
from graphrag.domain.versioning import GcReport, VersionInfo

VERSION = VersionInfo(3, datetime(2026, 11, 18, tzinfo=UTC), 12)


class RecordingStore:
    """Stands in for Neo4jStore: remembers how it was opened and what was asked of it."""

    opened: ClassVar[list[tuple[str, str, str]]] = []
    rollback_error: ClassVar[LookupError | None] = None
    gc_error: ClassVar[Exception | None] = None
    gc_calls: ClassVar[list[tuple[int, int, bool]]] = []

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

    def collect_garbage(self, *, keep: int, min_age_days: int, apply: bool) -> GcReport:
        RecordingStore.gc_calls.append((keep, min_age_days, apply))
        if RecordingStore.gc_error is not None:
            raise RecordingStore.gc_error
        return GcReport((3, 2), (1,), 5, applied=apply)


class StubEmbedder:
    dimensions = 2


@pytest.fixture(autouse=True)
def stubbed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    RecordingStore.opened = []
    RecordingStore.rollback_error = None
    RecordingStore.gc_error = None
    RecordingStore.gc_calls = []
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


def test_gc_is_a_dry_run_that_does_not_load_the_embedding_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden() -> None:
        raise AssertionError("collecting garbage must not load the embedding model")

    monkeypatch.setattr(index_cli, "load_default_embedder", forbidden)
    assert index_cli.main(["--gc"]) == 0
    assert RecordingStore.opened == [("bolt://db", "neo", "secret")]
    assert RecordingStore.gc_calls == [(3, 0, False)]


def test_gc_passes_its_options_and_deletes_only_with_apply() -> None:
    assert index_cli.main(["--gc", "--keep", "5", "--min-age-days", "14", "--apply"]) == 0
    assert RecordingStore.gc_calls == [(5, 14, True)]


@pytest.mark.parametrize("error", [LookupError("no version"), ValueError("keep at least 2")])
def test_gc_refused_by_the_store_exits_with_2(
    error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    RecordingStore.gc_error = error
    lines: list[str] = []
    monkeypatch.setattr(index_cli, "log_line", lines.append)
    assert index_cli.main(["--gc", "--apply"]) == 2
    assert lines == [f"gc: {error}"]
