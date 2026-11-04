import json
from datetime import UTC, datetime

import pytest

from graphrag.cli.evaluate import (
    build_report,
    format_differences,
    format_table,
    render,
    select_questions,
)
from graphrag.domain.evaluation import EvalQuestion, split_of
from graphrag.domain.models import SearchHit
from graphrag.pipeline.evaluation import (
    evaluate_retrieval,
    paired_differences,
    summarize_results,
)


def q(qid: str, kind: str, *articles: str, validated: bool = True) -> EvalQuestion:
    return EvalQuestion(qid, kind, f"{qid}?", "a", articles, {}, validated)  # type: ignore[arg-type]


class Scripted:
    def __init__(self, script: dict[str, list[str]]) -> None:
        self.script = script

    def retrieve(self, question: str, k: int) -> list[SearchHit]:
        return [SearchHit(a, "t", 0, "x", 0.0) for a in self.script[question]][:k]


QUESTIONS = [q("f1", "factual", "A"), q("f2", "factual", "B"), q("c1", "comparative", "C", "D")]
KS = (1, 3)
RESULTS = evaluate_retrieval(
    QUESTIONS,
    {
        "vector": Scripted({"f1?": ["x", "A"], "f2?": ["x"], "c1?": ["C", "x"]}),
        "hybrid": Scripted({"f1?": ["A"], "f2?": ["B"], "c1?": ["C", "D"]}),
    },
    KS,
    clock=lambda: 0.0,
)


def test_only_validated_questions_unless_asked() -> None:
    mixed = [q("a", "factual", "A"), q("b", "factual", "B", validated=False)]
    assert [x.id for x in select_questions(mixed, False)] == ["a"]
    assert [x.id for x in select_questions(mixed, True)] == ["a", "b"]


def test_table_layout() -> None:
    table = format_table("vector", summarize_results(RESULTS["vector"], KS), KS)
    assert table.splitlines() == [
        "== vector ==",
        "group          n    R@1    R@3    MRR  nDCG@10    s/q",
        "all            3   0.17   0.50   0.50     0.41  0.000",
        "factual        2   0.00   0.50   0.25     0.32  0.000",
        "comparative    1   0.50   0.50   1.00     0.61  0.000",
    ]


def test_differences_show_the_interval_and_flag_significance() -> None:
    diffs = paired_differences(RESULTS["vector"], RESULTS["hybrid"], KS)
    sizes = {"all": 3, "factual": 2, "comparative": 1}
    text = format_differences("vector", "hybrid", diffs, sizes, KS)
    lines = text.splitlines()
    assert lines[0].startswith("== hybrid minus vector: paired, 95% bootstrap interval")
    assert "recall@3" in lines[1]  # no k=5, so the largest k is shown
    assert lines[2].startswith("all")
    assert "+0.50 [" in lines[2]  # per-question MRR gaps: +0.5, +1, 0
    assert lines[-1].startswith("* = the interval excludes 0")


def test_a_gap_whose_interval_excludes_zero_is_starred() -> None:
    always_better = [q(f"f{i}", "factual", "A") for i in range(12)]
    results = evaluate_retrieval(
        always_better,
        {
            "base": Scripted({f"f{i}?": ["x"] for i in range(12)}),
            "better": Scripted({f"f{i}?": ["A"] for i in range(12)}),
        },
        (1,),
        clock=lambda: 0.0,
    )
    diffs = paired_differences(results["base"], results["better"], (1,))
    text = format_differences("base", "better", diffs, {"all": 12, "factual": 12}, (1,))
    assert "+1.00 [+1.00, +1.00]*" in text


def test_no_star_under_ten_questions_even_if_the_interval_excludes_zero() -> None:
    def gap(n: int) -> str:
        questions = [q(f"f{i}", "factual", "A") for i in range(n)]
        results = evaluate_retrieval(
            questions,
            {
                "base": Scripted({f"f{i}?": ["x"] for i in range(n)}),
                "better": Scripted({f"f{i}?": ["A"] for i in range(n)}),
            },
            (1,),
            clock=lambda: 0.0,
        )
        diffs = paired_differences(results["base"], results["better"], (1,))
        return format_differences("base", "better", diffs, {"all": n, "factual": n}, (1,))

    assert "+1.00 [+1.00, +1.00]*" in gap(10)
    assert "+1.00 [+1.00, +1.00] " in gap(9)
    assert "*" not in gap(9).splitlines()[2]
    assert "Never starred under 10 questions" in gap(9)


def test_a_significantly_worse_configuration_is_starred_too() -> None:
    questions = [q(f"f{i}", "factual", "A") for i in range(12)]
    results = evaluate_retrieval(
        questions,
        {
            "base": Scripted({f"f{i}?": ["A"] for i in range(12)}),
            "worse": Scripted({f"f{i}?": ["x"] for i in range(12)}),
        },
        (1,),
        clock=lambda: 0.0,
    )
    diffs = paired_differences(results["base"], results["worse"], (1,))
    text = format_differences("base", "worse", diffs, {"all": 12, "factual": 12}, (1,))
    assert "-1.00 [-1.00, -1.00]*" in text


def test_a_noisy_gap_is_not_starred() -> None:
    mixed = [q(f"f{i}", "factual", "A") for i in range(4)]
    results = evaluate_retrieval(
        mixed,
        {
            "base": Scripted({"f0?": ["A"], "f1?": ["x"], "f2?": ["A"], "f3?": ["x"]}),
            "other": Scripted({"f0?": ["x"], "f1?": ["A"], "f2?": ["A"], "f3?": ["x"]}),
        },
        (1,),
        clock=lambda: 0.0,
    )
    diffs = paired_differences(results["base"], results["other"], (1,))
    text = format_differences("base", "other", diffs, {"all": 4, "factual": 4}, (1,))
    row = text.splitlines()[2]
    assert "+0.00 [-" in row  # mean gap 0, interval straddling 0
    assert "*" not in row


def test_render_warns_about_small_samples_and_provisional_runs() -> None:
    text = render(QUESTIONS, RESULTS, KS, include_unvalidated=True)
    assert text.startswith("PROVISIONAL: unvalidated questions are included.")
    assert "3 questions" in text
    assert "Only 3 questions" in text
    clean = render(QUESTIONS, RESULTS, KS, include_unvalidated=False)
    assert "PROVISIONAL" not in clean


def test_report_is_json_with_settings_for_reproducibility() -> None:
    now = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
    report = build_report(QUESTIONS, RESULTS, KS, include_unvalidated=False, now=now)
    decoded = json.loads(json.dumps(report))
    assert decoded["created"] == "2026-11-04T12:00:00+00:00"
    assert decoded["provisional"] is False
    assert decoded["questions"] == 3
    assert decoded["ks"] == [1, 3]
    assert decoded["settings"]["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert decoded["settings"]["rrf_k"] == 60
    assert set(decoded["configs"]) == {"vector", "hybrid"}
    overall = decoded["configs"]["hybrid"]["summary"]["all"]
    assert overall["n"] == 3
    assert overall["recall"]["1"]["mean"] == pytest.approx(5 / 6)
    assert overall["mrr"]["mean"] == pytest.approx(1.0)
    per_question = decoded["configs"]["vector"]["questions"][0]
    assert per_question["question_id"] == "f1"
    assert per_question["retrieved"] == ["x", "A"]
    assert per_question["recall"] == {"1": 0.0, "3": 1.0}
    assert "hybrid minus vector" in decoded["differences"]
    assert decoded["differences"]["hybrid minus vector"]["all"]["mrr"]["n"] == 3


def test_a_split_keeps_only_that_half_and_questions_never_move() -> None:
    questions = [q(f"q{i}", "factual", "A") for i in range(40)]
    tune = select_questions(questions, False, "tune")
    report = select_questions(questions, False, "report")
    assert {x.id for x in tune} | {x.id for x in report} == {x.id for x in questions}
    assert not {x.id for x in tune} & {x.id for x in report}
    assert all(split_of(x.id) == "tune" for x in tune)
    assert all(split_of(x.id) == "report" for x in report)
    assert tune
    assert report
    assert {x.id for x in select_questions(questions[:20], False, "tune")} <= {x.id for x in tune}
    assert len(select_questions(questions, False)) == 40


def test_the_split_applies_on_top_of_validation() -> None:
    mixed = [q(f"q{i}", "factual", "A", validated=i % 2 == 0) for i in range(40)]
    assert all(x.validated for x in select_questions(mixed, False, "tune"))
    assert any(not x.validated for x in select_questions(mixed, True, "tune"))


def test_differences_state_the_smallest_gap_that_can_be_shown() -> None:
    questions = [q(f"f{i}", "factual", "A") for i in range(12)]
    script_a = {f"f{i}?": (["A"] if i % 2 else ["x"]) for i in range(12)}
    script_b = {f"f{i}?": (["A"] if i % 3 else ["x"]) for i in range(12)}
    results = evaluate_retrieval(
        questions, {"a": Scripted(script_a), "b": Scripted(script_b)}, (1,), clock=lambda: 0.0
    )
    diffs = paired_differences(results["a"], results["b"], (1,))
    text = format_differences("a", "b", diffs, {"all": 12, "factual": 12}, (1,))
    line = next(x for x in text.splitlines() if x.startswith("Smallest recall@1 gap"))
    gap = diffs["all"]["recall@1"].detectable_gap
    assert gap is not None
    assert f"all ±{gap:.2f}" in line
    assert "factual ±" in line
    assert "cannot be told from chance" in line


def test_no_detectable_gap_line_with_a_single_question() -> None:
    results = evaluate_retrieval(
        [q("f1", "factual", "A")],
        {"a": Scripted({"f1?": ["A"]}), "b": Scripted({"f1?": ["x"]})},
        (1,),
        clock=lambda: 0.0,
    )
    diffs = paired_differences(results["a"], results["b"], (1,))
    text = format_differences("a", "b", diffs, {"all": 1, "factual": 1}, (1,))
    assert "Smallest" not in text


def test_the_report_names_the_split_and_the_render_warns_about_it() -> None:
    results = evaluate_retrieval(
        QUESTIONS,
        {"vector": Scripted({"f1?": ["A"], "f2?": ["B"], "c1?": ["C", "D"]})},
        KS,
        clock=lambda: 0.0,
    )
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert (
        build_report(QUESTIONS, results, KS, include_unvalidated=False, now=now)["split"] == "all"
    )
    tuned = build_report(QUESTIONS, results, KS, include_unvalidated=False, now=now, split="tune")
    assert tuned["split"] == "tune"
    assert "TUNING HALF" in render(QUESTIONS, results, KS, include_unvalidated=False, split="tune")
    assert "REPORT HALF" in render(
        QUESTIONS, results, KS, include_unvalidated=False, split="report"
    )
    assert "HALF" not in render(QUESTIONS, results, KS, include_unvalidated=False)


def test_the_json_report_carries_the_detectable_gap() -> None:
    questions = [q(f"f{i}", "factual", "A") for i in range(6)]
    results = evaluate_retrieval(
        questions,
        {
            "a": Scripted({f"f{i}?": (["A"] if i % 2 else ["x"]) for i in range(6)}),
            "b": Scripted({f"f{i}?": ["A"] for i in range(6)}),
        },
        (1,),
        clock=lambda: 0.0,
    )
    report = build_report(
        questions, results, (1,), include_unvalidated=False, now=datetime(2026, 1, 1, tzinfo=UTC)
    )
    entry = report["differences"]["b minus a"]["all"]["recall@1"]
    assert entry["detectable_gap"] is not None
    assert entry["stdev"] > 0
