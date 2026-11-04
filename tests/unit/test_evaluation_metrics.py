import math
import statistics

import pytest

from graphrag.domain.evaluation import (
    QUESTION_TYPES,
    SPLITS,
    EvalQuestion,
    check_unique_ids,
    ndcg_at_k,
    parse_question,
    recall_at_k,
    reciprocal_rank,
    split_of,
    summarize,
)


def groups(*ids: str) -> list[frozenset[str]]:
    return [frozenset({i}) for i in ids]


def test_question_types() -> None:
    assert QUESTION_TYPES == ("factual", "relational", "comparative", "evolutive")


@pytest.mark.parametrize(
    ("retrieved", "k", "expected"),
    [
        (["X", "A", "Y"], 1, 0.0),
        (["X", "A", "Y"], 2, 0.5),  # only A of the two expected articles
        (["A", "B", "Y"], 1, 0.5),
        (["A", "B", "Y"], 2, 1.0),
        (["X", "Y", "Z"], 3, 0.0),
        ([], 5, 0.0),
        (["A", "A", "B"], 3, 1.0),  # a repeated id counts once
        (["Z", "Z", "B"], 2, 0.0),  # k counts positions, not distinct ids
    ],
)
def test_recall_at_k(retrieved: list[str], k: int, expected: float) -> None:
    assert recall_at_k(groups("A", "B"), retrieved, k) == pytest.approx(expected)


def test_recall_counts_each_expected_article_once() -> None:
    both_aliases = [frozenset({"A", "A2"}), frozenset({"B"})]
    assert recall_at_k(both_aliases, ["A", "A2", "X"], 3) == pytest.approx(0.5)


def test_recall_needs_expected_articles() -> None:
    with pytest.raises(ValueError, match="at least one"):
        recall_at_k([], ["A"], 1)


@pytest.mark.parametrize(
    ("retrieved", "expected"),
    [
        (["A"], 1.0),
        (["X", "A"], 0.5),
        (["X", "Y", "B", "A"], 1 / 3),
        (["X", "Y"], 0.0),
        ([], 0.0),
    ],
)
def test_reciprocal_rank_is_for_the_first_expected_article(
    retrieved: list[str], expected: float
) -> None:
    assert reciprocal_rank(groups("A", "B"), retrieved) == pytest.approx(expected)


def test_equivalent_ids_count_as_the_expected_article() -> None:
    aliased = [frozenset({"A", "A2"})]
    assert reciprocal_rank(aliased, ["X", "A2"]) == pytest.approx(0.5)
    assert recall_at_k(aliased, ["A2"], 1) == 1.0


def test_summarize_mean_and_interval() -> None:
    summary = summarize([1.0, 0.0, 1.0, 1.0], seed=1)
    assert summary.mean == pytest.approx(0.75)
    assert summary.n == 4
    assert 0.0 <= summary.low <= summary.mean <= summary.high <= 1.0
    assert summary.low < summary.high


def test_summarize_interval_has_the_expected_width() -> None:
    # 10 zeros and 10 ones: mean 0.5, standard error 0.5 / sqrt(20) = 0.112,
    # so a 95% interval is about 0.5 +/- 0.22
    summary = summarize([0.0] * 10 + [1.0] * 10)
    assert summary.mean == pytest.approx(0.5)
    assert 0.25 <= summary.low <= 0.35
    assert 0.65 <= summary.high <= 0.75


def test_summarize_is_reproducible_and_seeded() -> None:
    values = [float(i % 2) for i in range(20)]
    assert summarize(values, seed=3) == summarize(values, seed=3)


def test_summarize_constant_values_have_no_spread() -> None:
    summary = summarize([0.5] * 10)
    assert (summary.mean, summary.low, summary.high) == (0.5, 0.5, 0.5)


def test_summarize_edge_cases() -> None:
    assert summarize([]).n == 0
    one = summarize([0.25])
    assert (one.mean, one.low, one.high, one.n) == (0.25, 0.25, 0.25, 1)


def test_wider_interval_with_fewer_questions() -> None:
    few = summarize([1.0, 0.0] * 3)
    many = summarize([1.0, 0.0] * 50)
    assert (few.high - few.low) > (many.high - many.low)


def raw(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "q1",
        "type": "factual",
        "question": " What? ",
        "expected_answer": "That.",
        "expected_articles": ["2401.00001v2"],
    }
    base.update(overrides)
    return base


def test_parse_question() -> None:
    question = parse_question(raw())
    assert question == EvalQuestion("q1", "factual", "What?", "That.", ("2401.00001",), {}, False)
    assert question.groups == [frozenset({"2401.00001"})]


def test_parse_question_with_equivalents_and_validation() -> None:
    question = parse_question(
        raw(
            type="comparative",
            expected_articles=["2401.00001", "2401.00002"],
            equivalents={"2401.00002": ["2402.00009"]},
            validated=True,
        )
    )
    assert question.validated is True
    assert question.groups == [frozenset({"2401.00001"}), frozenset({"2401.00002", "2402.00009"})]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"id": ""}, "id must be a non-empty text"),
        ({"type": "weird"}, "type must be one of"),
        ({"question": "  "}, "question must be a non-empty text"),
        ({"expected_answer": None}, "expected_answer must be a non-empty text"),
        ({"expected_articles": []}, "expected_articles is empty"),
        ({"expected_articles": "2401.00001"}, "must be a list"),
        ({"expected_articles": ["nonsense"]}, "not an arXiv identifier"),
        ({"expected_articles": ["2401.00001", "2401.00001v2"]}, "duplicates"),
        ({"type": "relational"}, "needs at least 2 expected articles"),
        ({"type": "evolutive"}, "needs at least 2 expected articles"),
        ({"equivalents": {"2401.00009": ["2401.00003"]}}, "not an expected article"),
        ({"equivalents": ["2401.00003"]}, "equivalents must be an object"),
        ({"equivalents": {"2401.00001": ["bad"]}}, "not an arXiv identifier"),
        ({"validated": "yes"}, "validated must be true or false"),
    ],
)
def test_parse_question_rejects_inconsistent_data(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_question(raw(**overrides))


def test_duplicate_ids_are_rejected() -> None:
    first = parse_question(raw())
    check_unique_ids([first, parse_question(raw(id="q2"))])
    with pytest.raises(ValueError, match="duplicate question id q1"):
        check_unique_ids([first, first])


def test_style_defaults_to_named_and_is_validated() -> None:
    base = {
        "id": "q1",
        "type": "factual",
        "question": "What?",
        "expected_answer": "That.",
        "expected_articles": ["2401.00001"],
    }
    assert parse_question(base).style == "named"
    assert parse_question({**base, "style": "descriptive"}).style == "descriptive"
    with pytest.raises(ValueError, match="q1: style must be one of"):
        parse_question({**base, "style": "vague"})


# --- graded relevance: also_relevant, answerable, nDCG --------------------------------------

LOG3 = math.log2(3)


def test_ndcg_is_one_for_the_ideal_ranking() -> None:
    assert ndcg_at_k(groups("A", "B"), {"X"}, ["A", "B", "X"], 10) == pytest.approx(1.0)
    assert ndcg_at_k(groups("A"), set(), ["A"], 10) == pytest.approx(1.0)


def test_ndcg_hand_computed_with_an_also_relevant_article_ranked_before_an_expected_one() -> None:
    # retrieved A(2) X(1) B(2): dcg = 2 + 1/log2(3) + 2/2 ; ideal 2, 2, 1: 2 + 2/log2(3) + 1/2
    dcg = 2 + 1 / LOG3 + 1.0
    ideal = 2 + 2 / LOG3 + 0.5
    assert ndcg_at_k(groups("A", "B"), {"X"}, ["A", "X", "B"], 10) == pytest.approx(dcg / ideal)


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved() -> None:
    assert ndcg_at_k(groups("A"), {"X"}, ["Y", "Z"], 10) == 0.0


def test_ndcg_only_looks_at_the_first_k() -> None:
    assert ndcg_at_k(groups("A"), set(), ["Y", "A"], 1) == 0.0
    assert ndcg_at_k(groups("A"), set(), ["Y", "A"], 2) == pytest.approx(1 / LOG3)


def test_ndcg_counts_each_article_once_and_an_equivalent_id_as_its_article() -> None:
    twin = [frozenset({"A", "A2"})]
    assert ndcg_at_k(twin, set(), ["A", "A2"], 10) == pytest.approx(1.0)  # second twin: no gain
    assert ndcg_at_k(twin, set(), ["A2"], 10) == pytest.approx(1.0)
    assert ndcg_at_k(groups("A"), {"X"}, ["X", "X", "A"], 10) == pytest.approx(
        (1 + 2 / math.log2(4)) / (2 + 1 / LOG3)
    )


def test_ndcg_needs_expected_articles() -> None:
    with pytest.raises(ValueError, match="at least one expected article"):
        ndcg_at_k([], set(), ["A"], 10)


def test_graded_fields_default_to_nothing_and_answerable() -> None:
    question = parse_question(raw())
    assert question.also_relevant == ()
    assert question.judged_not_relevant == ()
    assert question.answerable is True


def test_parse_question_reads_judgements_and_normalises_ids() -> None:
    question = parse_question(
        raw(also_relevant=["2402.00001v3"], judged_not_relevant=["2403.00001"])
    )
    assert question.also_relevant == ("2402.00001",)
    assert question.judged_not_relevant == ("2403.00001",)


def test_an_unanswerable_question_has_no_expected_article() -> None:
    question = parse_question(raw(answerable=False, expected_articles=[]))
    assert question.answerable is False
    assert question.expected_articles == ()
    without_key = {k: v for k, v in raw(answerable=False).items() if k != "expected_articles"}
    assert parse_question(without_key).expected_articles == ()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"answerable": False}, "unanswerable question has no expected_articles"),
        ({"answerable": "no"}, "answerable must be true or false"),
        ({"also_relevant": ["2401.00001"]}, "cannot also be judged"),
        ({"judged_not_relevant": ["2401.00001"]}, "cannot also be judged"),
        (
            {"equivalents": {"2401.00001": ["2402.00009"]}, "also_relevant": ["2402.00009"]},
            "cannot also be judged",
        ),
        ({"also_relevant": ["2402.00001"], "judged_not_relevant": ["2402.00001"]}, "both relevant"),
        (
            {"answerable": False, "expected_articles": [], "also_relevant": ["2402.00001"]},
            "no relevant",
        ),
        ({"also_relevant": "2402.00001"}, "must be a list"),
    ],
)
def test_graded_fields_are_validated(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_question(raw(**overrides))


# --- detectable gap and the tune/report split ---------------------------------------------


def test_summary_carries_the_sample_standard_deviation() -> None:
    values = [0.0, 1.0, 0.0, 1.0]
    assert summarize(values).stdev == pytest.approx(statistics.stdev(values))
    assert summarize([0.5]).stdev == 0.0
    assert summarize([]).stdev == 0.0


def test_detectable_gap_is_about_two_point_eight_standard_errors() -> None:
    summary = summarize([0.0, 1.0, 0.0, 1.0])  # stdev 0.5774, n 4
    assert summary.detectable_gap == pytest.approx(2.8 * 0.5773502691896257 / 2)


def test_detectable_gap_shrinks_with_more_questions_and_is_zero_without_spread() -> None:
    few = summarize([0.0, 1.0] * 5)
    many = summarize([0.0, 1.0] * 50)
    assert few.detectable_gap is not None
    assert many.detectable_gap is not None
    assert many.detectable_gap < few.detectable_gap
    assert summarize([0.5, 0.5, 0.5]).detectable_gap == 0.0


def test_detectable_gap_is_unknown_below_two_questions() -> None:
    assert summarize([1.0]).detectable_gap is None
    assert summarize([]).detectable_gap is None


def test_split_is_stable_and_one_of_two_halves() -> None:
    assert split_of("factual-01") == split_of("factual-01")
    assert {split_of(f"q{i}") for i in range(50)} == set(SPLITS)


def test_split_is_roughly_balanced() -> None:
    tune = sum(split_of(f"q{i}") == "tune" for i in range(200))
    assert 80 <= tune <= 120


def test_split_depends_on_the_id_alone() -> None:
    # pinned values: changing the rule would silently move questions between the halves
    assert [split_of(i) for i in ("factual-01", "factual-02", "comparative-01")] == [
        "report",
        "report",
        "tune",
    ]
