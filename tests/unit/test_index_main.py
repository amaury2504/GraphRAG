from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import ClassVar, Self

import pytest

import graphrag.cli.index as index_cli
from graphrag.domain.models import IndexInfo
from graphrag.domain.versioning import GcReport, VersionInfo, VersionNotReadyError

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
    settings = SimpleNamespace(
        neo4j_uri="bolt://db",
        neo4j_user="neo",
        neo4j_password=secret,
        embedding_model="test/model",
    )
    monkeypatch.setattr(index_cli, "Settings", lambda: settings)
    monkeypatch.setattr(index_cli, "Neo4jStore", RecordingStore)
    monkeypatch.setattr(index_cli, "log_line", lambda _message: None)


def test_rollback_opens_the_store_and_rolls_back_without_loading_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(_name: str) -> None:
        raise AssertionError("rolling back must not load the embedding model")

    monkeypatch.setattr(index_cli, "load_embedder", forbidden)
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


def test_a_version_the_indexes_do_not_cover_exits_with_4_and_says_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> index_cli.IndexReport:
        raise VersionNotReadyError(5, "2 of 9 chunks are not in the vector index")

    lines: list[str] = []
    monkeypatch.setattr(index_cli, "log_line", lines.append)
    monkeypatch.setattr(index_cli, "load_embedder", lambda _name: StubEmbedder())
    monkeypatch.setattr(index_cli, "index_info_for", lambda _name, _dimensions: None)
    monkeypatch.setattr(index_cli, "run_indexing", refuse)
    assert index_cli.main([]) == 4
    assert lines[-1] == (
        "chunks: version 5 is not ready to be read: 2 of 9 chunks are not in the vector index; "
        "the active version is unchanged, run again to build a new one"
    )


def test_indexing_runs_with_the_embedder_and_prints_the_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: dict[str, object] = {}

    def fake_run(store: object, embedder: object, **kwargs: object) -> index_cli.IndexReport:
        calls.update(store=store, embedder=embedder, **kwargs)
        return index_cli.IndexReport(articles=1, chunks=2, total_chunks=3, version=4)

    info = IndexInfo("m", "r", 2, "title. chunk")
    names: list[str] = []

    def load(name: str) -> StubEmbedder:
        names.append(name)
        return StubEmbedder()

    monkeypatch.setattr(index_cli, "load_embedder", load)
    monkeypatch.setattr(index_cli, "index_info_for", lambda _name, _dimensions: info)
    monkeypatch.setattr(index_cli, "run_indexing", fake_run)
    assert index_cli.main(["--limit", "5", "--batch-size", "7"]) == 0
    assert (calls["limit"], calls["batch_size"], calls["info"]) == (5, 7, info)
    assert names == ["test/model"]  # the model comes from EMBEDDING_MODEL
    assert isinstance(calls["store"], RecordingStore)
    assert capsys.readouterr().out.strip() == (
        "IndexReport(articles=1, chunks=2, total_chunks=3, version=4)"
    )


def test_indexing_with_another_model_than_the_active_one_exits_with_2_and_says_how_to_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> index_cli.IndexReport:
        raise ValueError("the stored vectors were built differently: model: stored 'a'")

    lines: list[str] = []
    monkeypatch.setattr(index_cli, "log_line", lines.append)
    monkeypatch.setattr(index_cli, "load_embedder", lambda _name: StubEmbedder())
    monkeypatch.setattr(index_cli, "index_info_for", lambda _name, _dimensions: None)
    monkeypatch.setattr(index_cli, "run_indexing", refuse)
    assert index_cli.main([]) == 2
    assert lines[-1] == "chunks: the stored vectors were built differently: model: stored 'a'"


def test_reembed_switches_to_the_configured_model_without_indexing_articles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = IndexInfo("test/model", "r", 2, "title. chunk")
    calls: dict[str, object] = {}

    def fake_reembed(store: object, embedder: object, given: object, **kwargs: object) -> None:
        calls.update(store=store, embedder=embedder, info=given, **kwargs)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("re-embedding must not index articles")

    monkeypatch.setattr(index_cli, "load_embedder", lambda _name: StubEmbedder())
    monkeypatch.setattr(index_cli, "index_info_for", lambda _name, _dimensions: info)
    monkeypatch.setattr(index_cli, "run_reembed", fake_reembed)
    monkeypatch.setattr(index_cli, "run_indexing", forbidden)
    assert index_cli.main(["--reembed", "--batch-size", "9"]) == 0
    assert (calls["info"], calls["batch_size"]) == (info, 9)
    assert isinstance(calls["store"], RecordingStore)


@pytest.mark.parametrize(
    ("error", "code", "line"),
    [
        (LookupError("no version published"), 2, "reembed: no version published"),
        (
            VersionNotReadyError(6, "3 of 9 chunks are not in the vector index"),
            4,
            "reembed: version 6 is not ready to be read: 3 of 9 chunks are not in the vector "
            "index; the active version is unchanged, run again to resume",
        ),
    ],
)
def test_a_reembed_that_cannot_finish_exits_with_a_code_and_leaves_the_active_version(
    error: Exception, code: int, line: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise error

    lines: list[str] = []
    monkeypatch.setattr(index_cli, "log_line", lines.append)
    monkeypatch.setattr(index_cli, "load_embedder", lambda _name: StubEmbedder())
    monkeypatch.setattr(index_cli, "index_info_for", lambda _name, _dimensions: None)
    monkeypatch.setattr(index_cli, "run_reembed", refuse)
    assert index_cli.main(["--reembed"]) == code
    assert lines[-1] == line


def test_reembed_cannot_be_combined_with_rollback_or_gc() -> None:
    for other in ("--rollback", "--gc"):
        with pytest.raises(SystemExit):
            index_cli.main(["--reembed", other])


def test_gc_is_a_dry_run_that_does_not_load_the_embedding_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(_name: str) -> None:
        raise AssertionError("collecting garbage must not load the embedding model")

    monkeypatch.setattr(index_cli, "load_embedder", forbidden)
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
