import pytest

from graphrag.domain.evaluation import EvalQuestion
from graphrag.domain.models import SearchHit
from graphrag.pipeline.evaluation import (
    ALL,
    evaluate_retrieval,
    paired_differences,
    summarize_results,
)


def question(qid: str, kind: str, *articles: str, **equivalents: tuple[str, ...]) -> EvalQuestion:
    return EvalQuestion(
        qid,
        kind,  # type: ignore[arg-type]
        f"question {qid}?",
        "answer",
        articles,
        equivalents,
        True,
    )


QUESTIONS = [
    question("f1", "factual", "A"),
    question("f2", "factual", "B"),
    question("c1", "comparative", "C", "D"),
]


class ScriptedRetriever:
    """Returns the articles scripted for each question text, and records the depth asked."""

    def __init__(self, script: dict[str, list[str]]) -> None:
        self.script = script
        self.depths: list[int] = []

    def retrieve(self, question: str, k: int) -> list[SearchHit]:
        self.depths.append(k)
        return [SearchHit(a, f"t{a}", 0, "x", 0.0) for a in self.script[question]][:k]


GOOD = ScriptedRetriever(
    {"question f1?": ["A", "x"], "question f2?": ["x", "B"], "question c1?": ["C", "D"]}
)
WEAK = ScriptedRetriever(
    {"question f1?": ["x", "y", "A"], "question f2?": ["x"], "question c1?": ["D", "x"]}
)


def test_each_retriever_is_asked_every_question_at_the_deepest_k() -> None:
    retriever = ScriptedRetriever(GOOD.script)
    evaluate_retrieval(QUESTIONS, {"good": retriever}, ks=(1, 5))
    assert retriever.depths == [5, 5, 5]


def test_results_per_question() -> None:
    ticks = iter([0.0, 0.5, 1.0, 1.25, 2.0, 2.5])
    results = evaluate_retrieval(QUESTIONS, {"good": GOOD}, ks=(1, 2), clock=lambda: next(ticks))[
        "good"
    ]
    f1, f2, c1 = results
    assert (f1.question_id, f1.type, f1.retrieved) == ("f1", "factual", ("A", "x"))
    assert (f1.recall, f1.reciprocal_rank, f1.seconds) == ({1: 1.0, 2: 1.0}, 1.0, 0.5)
    assert (f2.recall, f2.reciprocal_rank, f2.seconds) == ({1: 0.0, 2: 1.0}, 0.5, 0.25)
    assert (c1.recall, c1.reciprocal_rank, c1.seconds) == ({1: 0.5, 2: 1.0}, 1.0, 0.5)


def test_equivalent_ids_are_credited() -> None:
    questions = [question("f1", "factual", "A", A=("A2",))]
    retriever = ScriptedRetriever({"question f1?": ["x", "A2"]})
    (result,) = evaluate_retrieval(questions, {"r": retriever}, ks=(2,))["r"]
    assert (result.recall[2], result.reciprocal_rank) == (1.0, 0.5)


@pytest.mark.parametrize("ks", [(), (0, 5), (-1,)])
def test_invalid_ks(ks: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="positive"):
        evaluate_retrieval(QUESTIONS, {"good": GOOD}, ks=ks)


def test_summary_is_overall_then_by_type_only_for_present_types() -> None:
    results = evaluate_retrieval(QUESTIONS, {"good": GOOD}, ks=(1, 3))["good"]
    summary = summarize_results(results, ks=(1, 3))
    assert list(summary) == [ALL, "factual", "comparative"]
    assert summary[ALL].n == 3
    assert summary["factual"].n == 2
    assert summary[ALL].recall[1].mean == pytest.approx((1 + 0 + 0.5) / 3)
    assert summary[ALL].recall[3].mean == pytest.approx(1.0)
    assert summary[ALL].mrr.mean == pytest.approx((1 + 0.5 + 1) / 3)
    assert summary["factual"].mrr.mean == pytest.approx(0.75)
    assert summary["comparative"].recall[1].mean == pytest.approx(0.5)


def test_summary_reports_the_mean_time_per_question() -> None:
    ticks = iter([0.0, 0.5, 1.0, 1.25, 2.0, 2.5])  # durations 0.5, 0.25, 0.5
    results = evaluate_retrieval(QUESTIONS, {"good": GOOD}, ks=(1,), clock=lambda: next(ticks))[
        "good"
    ]
    summary = summarize_results(results, ks=(1,))
    assert summary[ALL].mean_seconds == pytest.approx(1.25 / 3)
    assert summary["factual"].mean_seconds == pytest.approx(0.375)
    assert summary["comparative"].mean_seconds == pytest.approx(0.5)


def test_summary_of_nothing() -> None:
    summary = summarize_results([], ks=(1,))
    assert list(summary) == [ALL]
    assert summary[ALL].n == 0
    assert summary[ALL].mean_seconds == 0.0


def test_paired_differences() -> None:
    results = evaluate_retrieval(QUESTIONS, {"weak": WEAK, "good": GOOD}, ks=(1, 3))
    diffs = paired_differences(results["weak"], results["good"], ks=(1, 3))
    # MRR weak: f1 1/3, f2 0, c1 1 (D is first) ; good: 1, 1/2, 1
    assert diffs[ALL]["mrr"].mean == pytest.approx(((1 - 1 / 3) + 0.5 + 0.0) / 3)
    # recall@1 weak: 0, 0, 0.5 ; good: 1, 0, 0.5
    assert diffs[ALL]["recall@1"].mean == pytest.approx(1 / 3)
    assert diffs["factual"]["recall@1"].mean == pytest.approx(0.5)
    assert diffs["comparative"]["recall@1"].mean == pytest.approx(0.0)
    assert set(diffs) == {ALL, "factual", "comparative"}


def test_difference_of_a_configuration_with_itself_is_zero() -> None:
    results = evaluate_retrieval(QUESTIONS, {"good": GOOD}, ks=(1,))["good"]
    diffs = paired_differences(results, results, ks=(1,))
    assert diffs[ALL]["mrr"].mean == 0.0
    assert (diffs[ALL]["mrr"].low, diffs[ALL]["mrr"].high) == (0.0, 0.0)


def test_paired_differences_need_the_same_questions() -> None:
    results = evaluate_retrieval(QUESTIONS, {"good": GOOD}, ks=(1,))["good"]
    with pytest.raises(ValueError, match="same questions"):
        paired_differences(results, results[:2], ks=(1,))
