import math

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
    assert retriever.depths == [10, 10, 10]  # nDCG@10 always needs ten articles


def test_the_depth_follows_the_largest_k_beyond_ten() -> None:
    retriever = ScriptedRetriever(GOOD.script)
    evaluate_retrieval(QUESTIONS, {"good": retriever}, ks=(3, 20))
    assert retriever.depths == [20, 20, 20]


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


def styled(qid: str, style: str, *articles: str) -> EvalQuestion:
    return EvalQuestion(
        qid,
        "factual",
        f"question {qid}?",
        "a",
        articles,
        {},
        True,
        style,  # type: ignore[arg-type]
    )


STYLED = [styled("n1", "named", "A"), styled("d1", "descriptive", "B")]


def test_results_carry_the_style_and_summaries_split_by_style_when_both_occur() -> None:
    results = evaluate_retrieval(
        STYLED, {"r": ScriptedRetriever({"question n1?": ["A"], "question d1?": ["x"]})}, ks=(1,)
    )
    assert [r.style for r in results["r"]] == ["named", "descriptive"]
    summaries = summarize_results(results["r"], (1,))
    assert summaries["named"].mrr.mean == 1.0
    assert summaries["descriptive"].mrr.mean == 0.0
    assert summaries["named"].n == summaries["descriptive"].n == 1


def test_no_style_group_when_only_one_style_is_present() -> None:
    results = evaluate_retrieval(QUESTIONS, {"good": GOOD}, ks=(1,))
    assert set(summarize_results(results["good"], (1,))) == {ALL, "factual", "comparative"}


def test_paired_differences_are_also_split_by_style() -> None:
    script = {"question n1?": ["A"], "question d1?": ["x"]}
    better = {"question n1?": ["A"], "question d1?": ["B"]}
    results = evaluate_retrieval(
        STYLED, {"a": ScriptedRetriever(script), "b": ScriptedRetriever(better)}, ks=(1,)
    )
    diffs = paired_differences(results["a"], results["b"], (1,))
    assert diffs["descriptive"]["mrr"].mean == 1.0
    assert diffs["named"]["mrr"].mean == 0.0


def graded(qid: str, *articles: str, also: tuple[str, ...] = ()) -> EvalQuestion:
    return EvalQuestion(
        qid, "factual", f"question {qid}?", "a", articles, {}, True, also_relevant=also
    )


def test_ndcg_counts_the_also_relevant_articles_and_recall_does_not() -> None:
    question = graded("g1", "A", also=("X",))
    retriever = ScriptedRetriever({"question g1?": ["X", "A"]})
    (result,) = evaluate_retrieval([question], {"r": retriever}, ks=(1, 3))["r"]
    assert result.recall == {1: 0.0, 3: 1.0}  # X is not an expected article
    assert result.reciprocal_rank == 0.5
    expected = (1 / 1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert result.ndcg == pytest.approx(expected)


def test_unanswerable_questions_are_left_out_of_retrieval_metrics() -> None:
    nothing = EvalQuestion("u1", "factual", "question u1?", "a", (), {}, True, answerable=False)
    retriever = ScriptedRetriever({"question f1?": ["A"], "question u1?": ["Z"]})
    results = evaluate_retrieval([QUESTIONS[0], nothing], {"r": retriever}, ks=(1,))["r"]
    assert [r.question_id for r in results] == ["f1"]
    assert retriever.depths == [10]  # asked once, for the answerable question only


def test_paired_differences_include_ndcg() -> None:
    results = evaluate_retrieval(
        [graded("g1", "A", also=("X",))],
        {
            "a": ScriptedRetriever({"question g1?": ["A", "X"]}),
            "b": ScriptedRetriever({"question g1?": ["X", "A"]}),
        },
        ks=(1,),
    )
    diff = paired_differences(results["a"], results["b"], (1,))["all"]
    assert diff["ndcg@10"].mean < 0  # b ranks the expected article second
    assert diff["mrr"].mean == pytest.approx(-0.5)
