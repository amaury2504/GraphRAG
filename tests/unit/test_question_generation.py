import random
from collections.abc import Sequence
from datetime import date, timedelta

import pytest

from graphrag.domain.evaluation import QUESTION_TYPES, QuestionType, parse_question
from graphrag.domain.models import Article, Completion, SearchHit, Usage
from graphrag.pipeline.question_generation import (
    DESCRIPTIVE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    QuestionPlan,
    UnrelatedArticlesError,
    build_prompt,
    choose_group,
    find_title_leak,
    find_vague_wording,
    generate_candidates,
    parse_generation,
    plan_questions,
    title_groups,
    title_overlap,
    title_word_frequencies,
    topic_neighbours,
    write_question,
)

WORDS = " ".join(f"word{i}" for i in range(100))


def art(
    n: int, *, title: str | None = None, words: str = WORDS, days: int | None = None
) -> Article:
    published = date(2025, 1, 1) + timedelta(days=30 * n if days is None else days)
    return Article(f"2501.{n:05d}", title or f"Topic {n}", words, published, ("cs.CL",), ())


def number_of(text: str) -> int:
    return int(text.split()[1].rstrip(".,"))  # "Topic 7. ..." -> 7


class FakeEmbedder:
    dimensions = 1

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(number_of(t))] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("not used")


class FakeSearch:
    """Neighbours of article n by meaning are n+1.., by keywords n+2.. (ids wrap at `size`)."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.calls = 0

    def _hits(self, ids: list[int]) -> list[SearchHit]:
        return [SearchHit(f"2501.{i % self.size:05d}", "t", 0, "x", 0.0) for i in ids]

    def vector_search(self, vector: Sequence[float], k: int) -> list[SearchHit]:
        self.calls += 1
        n = int(vector[0])
        return self._hits([n, *range(n + 1, n + 1 + k)])  # starts with the article itself

    def keyword_search(self, text: str, k: int) -> list[SearchHit]:
        n = number_of(text)
        return self._hits(list(range(n + 2, n + 2 + k)))


CORPUS = [art(n) for n in range(40)]


def plan(
    corpus: list[Article] = CORPUS,
    *,
    per_type: int = 2,
    seed: int = 7,
    types: Sequence[QuestionType] = QUESTION_TYPES,
) -> list[QuestionPlan]:
    return plan_questions(
        corpus, FakeSearch(len(corpus)), FakeEmbedder(), per_type=per_type, seed=seed, types=types
    )


def test_topic_neighbours_alternate_meaning_and_keywords_without_duplicates_or_self() -> None:
    class Fixed:
        def vector_search(self, vector: Sequence[float], k: int) -> list[SearchHit]:
            return [SearchHit(i, "t", 0, "x", 0.0) for i in ("S", "B", "C", "D", "B")]

        def keyword_search(self, text: str, k: int) -> list[SearchHit]:
            return [SearchHit(i, "t", 0, "x", 0.0) for i in ("C", "E")]

    article = Article("S", "Topic 1", WORDS, date(2025, 1, 1), (), ())
    # rank pairs: (S, C) (B, E) (C, -) (D, -): the seed and repeats are dropped
    assert topic_neighbours(article, Fixed(), FakeEmbedder()) == ["C", "B", "E", "D"]


def test_two_questions_per_type_in_type_order() -> None:
    kinds = [p.type for p in plan()]
    assert kinds == ["factual"] * 2 + ["relational"] * 2 + ["comparative"] * 2 + ["evolutive"] * 2


def test_group_sizes_per_type() -> None:
    sizes = {p.type: len(p.articles) for p in plan()}
    assert sizes == {"factual": 1, "relational": 3, "comparative": 2, "evolutive": 3}


def test_an_article_is_never_used_twice() -> None:
    ids = [a.arxiv_id for p in plan() for a in p.articles]
    assert len(ids) == len(set(ids)) == 2 * (1 + 3 + 2 + 3)


def test_planning_is_reproducible_for_a_seed() -> None:
    def ids(plans: list[QuestionPlan]) -> list[list[str]]:
        return [[a.arxiv_id for a in p.articles] for p in plans]

    assert ids(plan(seed=3)) == ids(plan(seed=3))
    assert ids(plan(seed=3)) != ids(plan(seed=4))


def test_companions_are_topic_neighbours_of_the_seed() -> None:
    for chosen in plan():
        if chosen.type in ("relational", "comparative"):
            seed_n = int(chosen.articles[0].arxiv_id.split(".")[1])
            for other in chosen.articles[1:]:
                gap = (int(other.arxiv_id.split(".")[1]) - seed_n) % 40
                assert 1 <= gap <= 12


def test_evolutive_groups_are_oldest_first_and_span_months() -> None:
    for chosen in plan(per_type=3):
        if chosen.type == "evolutive":
            dates = [a.published for a in chosen.articles]
            assert dates == sorted(dates)
            assert (dates[-1] - dates[0]).days >= 90


def neighbours_with_far_one_at(position: int) -> list[Article]:
    """Ten neighbours dated the same day as the seed, except one a year later at `position`."""
    return [
        art(100 + i, days=400 if i == position else 0, title=f"Neighbour {i}") for i in range(1, 11)
    ]


def test_an_evolution_group_uses_the_eight_closest_neighbours_but_no_more() -> None:
    seed = art(1, days=0)
    rng = random.Random(0)
    inside = choose_group("evolutive", seed, neighbours_with_far_one_at(8), rng, 90)
    assert inside is not None
    assert [a.published.year for a in inside] == [2025, 2025, 2026]
    assert choose_group("evolutive", seed, neighbours_with_far_one_at(9), rng, 90) is None


def test_relational_and_comparative_groups_draw_from_the_six_closest() -> None:
    seed = art(1)
    neighbours = [art(100 + i, title=f"Neighbour {i}") for i in range(1, 11)]
    for _ in range(20):
        comparative = choose_group("comparative", seed, neighbours, random.Random(_), 90)
        relational = choose_group("relational", seed, neighbours, random.Random(_), 90)
        assert comparative is not None
        assert relational is not None
        assert comparative[0] == seed
        assert relational[0] == seed
        for other in (*comparative[1:], *relational[1:]):
            assert int(other.arxiv_id.split(".")[1]) - 100 <= 6
    assert choose_group("relational", seed, neighbours[:1], random.Random(0), 90) is None


def test_short_abstracts_are_never_chosen() -> None:
    short = [art(n, words="too short") if n % 2 else art(n) for n in range(40)]
    chosen = {a.arxiv_id for p in plan(short, per_type=1) for a in p.articles}
    assert chosen
    assert all(int(i.split(".")[1]) % 2 == 0 for i in chosen)


def test_the_same_work_deposited_twice_is_used_once_and_never_paired_with_itself() -> None:
    twins = [art(n, title="Topic 1" if n in (1, 2) else None) for n in range(40)]
    for chosen in plan(twins, per_type=4):
        ids = {a.arxiv_id for a in chosen.articles}
        assert not {"2501.00001", "2501.00002"} <= ids
    used = [a.arxiv_id for p in plan(twins, per_type=4) for a in p.articles]
    assert not ({"2501.00001", "2501.00002"} <= set(used))


def test_planning_fails_loudly_when_no_group_exists() -> None:
    same_day = [art(n, days=0) for n in range(40)]
    with pytest.raises(RuntimeError, match="evolutive"):
        plan(same_day, per_type=1)
    with pytest.raises(RuntimeError, match="factual"):
        plan(CORPUS[:3], per_type=5)


def test_prompt_numbers_articles_with_titles_dates_and_clean_abstracts() -> None:
    first = art(1, words="line one\n  line   two")
    second = art(2)
    prompt = build_prompt(QuestionPlan("comparative", (first, second)))
    assert prompt.startswith("Question type: comparative. Ask for a comparison")
    assert "[1] Topic 1 (2025-01-31)\nline one line two" in prompt
    assert "[2] Topic 2 (2025-03-02)\nword0 word1" in prompt


def test_evolutive_prompt_says_articles_are_oldest_first() -> None:
    prompt = build_prompt(QuestionPlan("evolutive", (art(1), art(2), art(3))))
    assert "listed from the earliest to the latest" in prompt


def test_system_prompt_demands_named_standalone_questions_json_and_allows_a_skip() -> None:
    assert "self-contained and NAME what it asks about" in SYSTEM_PROMPT
    assert '"these works", "the two approaches"' in SYSTEM_PROMPT
    assert '{"skip": ' in SYSTEM_PROMPT
    assert "JSON object only" in SYSTEM_PROMPT


@pytest.mark.parametrize(
    "reply",
    [
        '{"question": "Q?", "expected_answer": "A."}',
        'Sure!\n```json\n{"question": " Q? ", "expected_answer": "A. "}\n```\nHope it helps',
        '{"question": "Q?",\n "expected_answer": "A.", "extra": 1}',
    ],
)
def test_parse_generation_accepts_noisy_replies(reply: str) -> None:
    assert parse_generation(reply) == ("Q?", "A.")


def test_the_model_may_decline_an_unrelated_group() -> None:
    with pytest.raises(UnrelatedArticlesError, match="no specific shared subject: unrelated"):
        parse_generation('{"skip": "unrelated"}')


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ("no json here", "no JSON object"),
        ('{"question": "Q?", ', "no JSON object"),
        ('{"question": "Q?" "expected_answer": "A"}', "invalid JSON"),
        ('{"question": "", "expected_answer": "A"}', "no question"),
        ('{"question": 3, "expected_answer": "A"}', "no question"),
        ('{"question": "Q?"}', "no expected_answer"),
        ('{"question": "Q?", "expected_answer": "  "}', "no expected_answer"),
    ],
)
def test_parse_generation_rejects_bad_replies(reply: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_generation(reply)


class ScriptedLLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Completion:
        self.prompts.append((system, user))
        return Completion(self.replies.pop(0), Usage(10, 5))


GOOD = '{"question": "Why?", "expected_answer": "Because."}'


def test_candidates_are_valid_unvalidated_questions_with_review_aids() -> None:
    plans = [QuestionPlan("factual", (art(1),)), QuestionPlan("comparative", (art(2), art(3)))]
    llm = ScriptedLLM([GOOD, GOOD])
    records, failures = generate_candidates(plans, llm, title_groups(CORPUS), log=lambda _m: None)
    assert failures == []
    assert [r["id"] for r in records] == ["factual-01", "comparative-01"]
    assert llm.prompts[0] == (SYSTEM_PROMPT, build_prompt(plans[0]))
    second = records[1]
    assert second["expected_articles"] == ["2501.00002", "2501.00003"]
    assert second["source_titles"] == ["Topic 2", "Topic 3"]
    assert second["source_dates"] == ["2025-03-02", "2025-04-01"]
    assert second["validated"] is False
    for record in records:
        assert parse_question(record).validated is False


def test_ids_count_per_type_from_the_first_index() -> None:
    plans = [QuestionPlan("factual", (art(n),)) for n in (1, 2)]
    records, _ = generate_candidates(
        plans, ScriptedLLM([GOOD, GOOD]), title_groups(CORPUS), first_index=5, log=print
    )
    assert [r["id"] for r in records] == ["factual-05", "factual-06"]


def test_a_failed_generation_is_reported_and_skipped_without_using_an_id() -> None:
    plans = [QuestionPlan("factual", (art(n),)) for n in (1, 2, 3)]
    llm = ScriptedLLM([GOOD, "garbage", GOOD])
    messages: list[str] = []
    records, failures = generate_candidates(plans, llm, title_groups(CORPUS), log=messages.append)
    assert [r["id"] for r in records] == ["factual-01", "factual-02"]
    assert failures == ["factual on 2501.00002: no JSON object in the reply"]
    assert "failed: factual on 2501.00002: no JSON object in the reply" in messages


def test_a_twin_deposit_is_recorded_as_an_equivalent() -> None:
    twins = [art(1, title="Same Work"), art(2, title="same  work"), art(3)]
    plans = [QuestionPlan("comparative", (twins[0], twins[2]))]
    (record,), _ = generate_candidates(plans, ScriptedLLM([GOOD]), title_groups(twins), log=print)
    assert record["equivalents"] == {"2501.00001": ["2501.00002"]}
    assert parse_question(record).groups[0] == frozenset({"2501.00001", "2501.00002"})


def test_title_groups_ignore_case_and_spacing() -> None:
    groups = title_groups([art(1, title="A  B"), art(2, title="a b"), art(3, title="C")])
    assert groups == {"a b": ["2501.00001", "2501.00002"], "c": ["2501.00003"]}


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How do these works differ?", "these works"),
        ("How do those two approaches compare?", "those two approaches"),
        ("What do the two systems share?", "the two"),
        ("Why did the first approach fail?", "the first approach"),
        ("What is the second article about?", "the second article"),
        ("What does this paper propose?", "this paper"),
        ("Does the study report gains?", "the study"),
        ("Which of both approaches is faster?", "both approaches"),
        ("HOW DO THESE METHODS DIFFER?", "THESE METHODS"),
    ],
)
def test_vague_wording_is_found(question: str, expected: str) -> None:
    assert find_vague_wording(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "How does LiveRAG differ from query-agnostic RAG evaluation?",
        "What does the first-stage retriever of CodeTeam return?",
        "Which two datasets does GRaSP use?",
        "How do recent approaches to agent memory differ?",
        "What is the second step of the TRATE pipeline?",
        "",
    ],
)
def test_specific_questions_are_not_flagged(question: str) -> None:
    assert find_vague_wording(question) is None


VAGUE = '{"question": "How do the two approaches differ?", "expected_answer": "A."}'
NAMED = '{"question": "How does LiveRAG differ from CodeTeam?", "expected_answer": "B."}'
PAIR = QuestionPlan("comparative", (art(2), art(3)))


def test_a_vague_question_is_rewritten_once_with_feedback() -> None:
    llm = ScriptedLLM([VAGUE, NAMED])
    assert write_question(PAIR, llm) == ("How does LiveRAG differ from CodeTeam?", "B.")
    first, second = llm.prompts
    assert first == (SYSTEM_PROMPT, build_prompt(PAIR))
    assert second[0] == SYSTEM_PROMPT
    assert second[1].startswith(build_prompt(PAIR))
    assert "it says 'the two'" in second[1]
    assert "naming the specific" in second[1]


def test_a_question_still_vague_after_the_rewrite_fails() -> None:
    llm = ScriptedLLM([VAGUE, VAGUE, NAMED])
    with pytest.raises(ValueError, match="still vague after a rewrite: 'the two'"):
        write_question(PAIR, llm)
    assert len(llm.prompts) == 2  # one rewrite, not more


def test_a_good_question_costs_one_call() -> None:
    llm = ScriptedLLM([NAMED])
    write_question(PAIR, llm)
    assert len(llm.prompts) == 1


def test_a_declined_group_is_not_retried() -> None:
    llm = ScriptedLLM(['{"skip": "nothing in common"}', NAMED])
    with pytest.raises(UnrelatedArticlesError):
        write_question(PAIR, llm)
    assert len(llm.prompts) == 1


def test_candidates_report_vague_and_declined_groups_and_keep_the_rest() -> None:
    plans = [
        PAIR,
        QuestionPlan("factual", (art(5),)),
        QuestionPlan("comparative", (art(6), art(7))),
    ]
    llm = ScriptedLLM([VAGUE, VAGUE, GOOD, '{"skip": "unrelated"}'])
    records, failures = generate_candidates(plans, llm, title_groups(CORPUS), log=lambda _m: None)
    assert [r["id"] for r in records] == ["factual-01"]
    assert failures == [
        "comparative on 2501.00002, 2501.00003: still vague after a rewrite: 'the two'",
        "comparative on 2501.00006, 2501.00007: no specific shared subject: unrelated",
    ]
    assert len(llm.prompts) == 4


def test_excluded_articles_and_their_twins_are_never_chosen() -> None:
    twins = [art(n, title="Topic 1" if n in (1, 2) else None) for n in range(40)]
    excluded = {"2501.00001", *(f"2501.{n:05d}" for n in range(10, 20))}
    plans = plan_questions(
        twins, FakeSearch(40), FakeEmbedder(), per_type=3, seed=5, exclude=excluded
    )
    chosen = {a.arxiv_id for p in plans for a in p.articles}
    assert chosen
    assert not chosen & excluded
    assert "2501.00002" not in chosen  # twin of an excluded article


def test_unknown_excluded_ids_are_ignored() -> None:
    assert plan(per_type=1) == plan_questions(
        CORPUS, FakeSearch(40), FakeEmbedder(), per_type=1, seed=7, exclude={"9999.99999"}
    )


def test_prompt_asks_to_flag_inferences_and_incomparable_figures() -> None:
    assert "the abstracts do not" in SYSTEM_PROMPT
    assert "whether they measure the\n  same thing" in SYSTEM_PROMPT


def test_planning_can_be_limited_to_some_types() -> None:
    plans = plan(per_type=2, types=("comparative", "evolutive"))
    assert [p.type for p in plans] == ["comparative"] * 2 + ["evolutive"] * 2


# --- descriptive style and title-leak detector ---------------------------------------------

TITLES = [
    "ARIA: Adaptive Retrieval for Agents",
    "Retrieval for Agents in Practice",
    "Adaptive Retrieval Methods",
    "Retrieval Evaluation Survey",
]
TITLE_ARTICLES = [art(n, title=t) for n, t in enumerate(TITLES, start=1)]
FREQ = title_word_frequencies(TITLE_ARTICLES)


def test_title_word_frequencies_count_distinct_titles_and_skip_function_words() -> None:
    assert FREQ["retrieval"] == 4
    assert FREQ["aria"] == 1
    assert FREQ["agents"] == 2
    assert "for" not in FREQ
    assert "in" not in FREQ


def test_twin_titles_count_once_in_the_frequencies() -> None:
    twins = [art(1, title="Zeta Method"), art(2, title="zeta  method")]
    assert title_word_frequencies(twins)["zeta"] == 1


def test_a_rare_title_word_is_a_leak() -> None:
    assert find_title_leak("How does ARIA choose what to fetch?", TITLES[:1], FREQ) == "aria"


def test_a_common_title_word_is_not_a_leak() -> None:
    assert find_title_leak("How does retrieval adapt to its inputs?", TITLES[:1], FREQ) is None


def test_the_frequency_limit_is_inclusive() -> None:
    assert find_title_leak("agents", TITLES[:1], FREQ, max_frequency=2) == "agents"
    assert find_title_leak("agents", TITLES[:1], FREQ, max_frequency=1) is None


def test_only_words_of_the_expected_titles_count() -> None:
    # "aria" is rare but belongs to another article's title: not a leak for this question
    assert find_title_leak("What is aria?", TITLES[1:], FREQ) is None


def test_a_word_missing_from_the_frequencies_counts_as_rare() -> None:
    assert find_title_leak("zorblax", ["Zorblax"], {}) == "zorblax"


def test_the_first_leak_in_alphabetical_order_is_reported() -> None:
    assert find_title_leak("zeta and alpha", ["Zeta Alpha"], {}) == "alpha"


def test_title_overlap_is_the_share_of_question_words_found_in_titles() -> None:
    # content words: aria, choose, fetch -> only "aria" is in the title
    assert title_overlap("How does ARIA choose what to fetch?", TITLES[:1]) == 0.333
    assert title_overlap("", TITLES[:1]) == 0.0
    assert title_overlap("anything", []) == 0.0


LEAKY = '{"question": "How does ARIA decide when to retrieve?", "expected_answer": "A."}'
CLEAN = '{"question": "How can an agent decide when to retrieve?", "expected_answer": "B."}'
DESCRIPTIVE = QuestionPlan("factual", (TITLE_ARTICLES[0],), "descriptive")


def test_a_descriptive_plan_uses_the_descriptive_prompt() -> None:
    llm = ScriptedLLM([CLEAN])
    write_question(DESCRIPTIVE, llm, FREQ)
    assert llm.prompts[0][0] == DESCRIPTIVE_SYSTEM_PROMPT
    assert DESCRIPTIVE_SYSTEM_PROMPT != SYSTEM_PROMPT
    assert "Do NOT use the name" in DESCRIPTIVE_SYSTEM_PROMPT
    assert "Do NOT use the name" not in SYSTEM_PROMPT


def test_both_prompts_share_the_answer_rules() -> None:
    for prompt in (SYSTEM_PROMPT, DESCRIPTIVE_SYSTEM_PROMPT):
        assert "the abstracts do not" in prompt
        assert '{"skip"' in prompt


def test_a_leaking_descriptive_question_is_rewritten_once_with_the_word() -> None:
    llm = ScriptedLLM([LEAKY, CLEAN])
    question, _ = write_question(DESCRIPTIVE, llm, FREQ)
    assert question == "How can an agent decide when to retrieve?"
    assert "'aria'" in llm.prompts[1][1]
    assert "Your previous question was rejected" in llm.prompts[1][1]


def test_a_question_still_leaking_after_the_rewrite_fails() -> None:
    llm = ScriptedLLM([LEAKY, LEAKY, CLEAN])
    with pytest.raises(ValueError, match="still names the article after a rewrite: 'aria'"):
        write_question(DESCRIPTIVE, llm, FREQ)
    assert len(llm.prompts) == 2


def test_a_named_question_may_use_the_names() -> None:
    llm = ScriptedLLM([LEAKY])
    named = QuestionPlan("factual", (TITLE_ARTICLES[0],), "named")
    assert write_question(named, llm, FREQ)[0].startswith("How does ARIA")
    assert len(llm.prompts) == 1


def test_a_descriptive_plan_without_frequencies_is_refused_before_any_call() -> None:
    llm = ScriptedLLM([CLEAN])
    with pytest.raises(ValueError, match="needs the title word frequencies"):
        write_question(DESCRIPTIVE, llm)
    assert llm.prompts == []


def test_a_vague_descriptive_question_is_told_to_describe() -> None:
    llm = ScriptedLLM([VAGUE, CLEAN])
    write_question(QuestionPlan("comparative", (art(2), art(3)), "descriptive"), llm, FREQ)
    assert "describing the specific problem" in llm.prompts[1][1]


def test_candidates_record_style_and_overlap() -> None:
    plans = [DESCRIPTIVE, QuestionPlan("factual", (TITLE_ARTICLES[1],))]
    records, failures = generate_candidates(
        plans,
        ScriptedLLM([CLEAN, GOOD]),
        title_groups(TITLE_ARTICLES),
        frequencies=FREQ,
        log=lambda _m: None,
    )
    assert failures == []
    assert [r["style"] for r in records] == ["descriptive", "named"]
    assert records[0]["title_overlap"] == 0.0  # no content word of the question is in the title
    assert all(parse_question(r).style == r["style"] for r in records)


def test_planning_passes_the_style_to_every_plan() -> None:
    plans = plan_questions(
        CORPUS, FakeSearch(40), FakeEmbedder(), per_type=2, seed=7, style="descriptive"
    )
    assert {p.style for p in plans} == {"descriptive"}
    assert {p.style for p in plan(per_type=1)} == {"named"}
