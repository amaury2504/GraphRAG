import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from graphrag.cli import sweep as cli
from graphrag.domain.chunking import ChunkerConfig
from graphrag.domain.evaluation import EvalQuestion
from graphrag.domain.models import SearchHit
from graphrag.pipeline.evaluation import evaluate_retrieval
from graphrag.pipeline.sweep import BASELINE, SweepPoint


class Fixed:
    def retrieve(self, question: str, k: int) -> list[SearchHit]:
        return [SearchHit("A", "t", 0, "x", 0.0)][:k]


QUESTIONS = [EvalQuestion("f1", "factual", "f1?", "a", ("A",), {}, True)]
KS = (1, 5)


def points() -> list[SweepPoint]:
    results = evaluate_retrieval(QUESTIONS, {"vector": Fixed(), "hybrid": Fixed()}, KS)
    return [SweepPoint(BASELINE, 10, results), SweepPoint(ChunkerConfig(100, 0), 20, results)]


def test_the_real_database_is_refused_as_a_target() -> None:
    with pytest.raises(cli.SameDatabaseError, match="throwaway"):
        cli.check_target("bolt://localhost:7687", "bolt://localhost:7687/")
    cli.check_target("bolt://localhost:7687", "bolt://localhost:7688")


def test_grid_defaults_and_overrides() -> None:
    assert len(cli.grid_from(None, None)) == 8
    grid = cli.grid_from([50, 200], [1])
    assert grid == [BASELINE, ChunkerConfig(50, 1)]
    only_overlaps = cli.grid_from(None, [0])
    assert only_overlaps[0] == BASELINE
    assert {c.overlap_sentences for c in only_overlaps[1:]} == {0}


def test_render_warns_about_a_small_question_set_and_the_split() -> None:
    text = cli.render(points(), QUESTIONS, KS, include_unvalidated=True, split="tune")
    assert "PROVISIONAL" in text
    assert "TUNING HALF" in text
    assert "Only 1 questions" in text
    assert "w100/o0" in text


def test_render_marks_the_report_half() -> None:
    text = cli.render(points(), QUESTIONS, KS, include_unvalidated=False, split="report")
    assert "REPORT HALF" in text
    assert "PROVISIONAL" not in text


def test_report_lists_every_configuration_with_its_figures() -> None:
    report = cli.build_report(
        points(),
        QUESTIONS,
        KS,
        split="tune",
        include_unvalidated=False,
        articles=3,
        embedding=("m", "r"),
        now=datetime(2026, 11, 20, tzinfo=UTC),
    )
    assert report["kind"] == "chunk_sweep"
    assert [p["label"] for p in report["points"]] == ["w200/o1", "w100/o0"]
    assert report["points"][1]["summary"]["vector"]["mrr"] == 1.0
    assert report["points"][1]["summary"]["hybrid"]["recall"]["5"] == 1.0
    json.dumps(report)  # serialisable


def test_main_refuses_to_index_the_real_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    questions = tmp_path / "q.jsonl"
    questions.write_text(
        json.dumps(
            {
                "id": "f1",
                "type": "factual",
                "question": "q?",
                "expected_answer": "a",
                "expected_articles": ["2401.00001"],
                "equivalents": {},
                "validated": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    code = cli.main(["--questions", str(questions), "--target-uri", "bolt://localhost:7687"])
    assert code == 2
    assert "throwaway" in capsys.readouterr().out


def test_main_stops_without_validated_questions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "q.jsonl"
    empty.write_text("", encoding="utf-8")
    assert cli.main(["--questions", str(empty)]) == 1
    assert "No validated question" in capsys.readouterr().out
