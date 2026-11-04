"""Evaluation questions and retrieval metrics (recall@k, MRR) with bootstrap intervals."""

import math
import random
import statistics
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from graphrag.domain.identifiers import normalize_arxiv_id

QuestionType = Literal["factual", "relational", "comparative", "evolutive"]
QUESTION_TYPES: tuple[QuestionType, ...] = get_args(QuestionType)

# "named" questions use the names of the systems (easy for keyword search); "descriptive"
# ones describe the problem the way a reader who has not seen the article would.
QuestionStyle = Literal["named", "descriptive"]
QUESTION_STYLES: tuple[QuestionStyle, ...] = get_args(QuestionStyle)


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    """A question with the articles that answer it.

    `equivalents` maps an expected article to other arXiv ids of the same work (a paper
    deposited twice): finding any of them counts as finding the article.

    The corpus only talks about RAG, so other articles than the expected ones can answer too.
    `also_relevant` lists those a person judged relevant when pooling (they count in nDCG, with
    a lower grade, and never in recall or MRR); `judged_not_relevant` lists those judged not
    relevant, so a later pooling does not ask again. An `answerable=False` question has no
    expected article: it checks that the system says it cannot answer.
    """

    id: str
    type: QuestionType
    question: str
    expected_answer: str
    expected_articles: tuple[str, ...]
    equivalents: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    validated: bool = False
    style: QuestionStyle = "named"
    also_relevant: tuple[str, ...] = ()
    judged_not_relevant: tuple[str, ...] = ()
    answerable: bool = True

    @property
    def groups(self) -> list[frozenset[str]]:
        """One set of acceptable ids per expected article."""
        return [
            frozenset((article, *self.equivalents.get(article, ())))
            for article in self.expected_articles
        ]


def parse_question(data: Mapping[str, Any]) -> EvalQuestion:
    """Build a question from a decoded JSON object, rejecting anything inconsistent."""
    qid = _text(data, "id")
    kind = _text(data, "type")
    if kind not in QUESTION_TYPES:
        raise ValueError(f"{qid}: type must be one of {QUESTION_TYPES}, got {kind!r}")
    answerable = data.get("answerable", True)
    if not isinstance(answerable, bool):
        raise ValueError(f"{qid}: answerable must be true or false")
    articles = _ids(qid, data.get("expected_articles", []), "expected_articles")
    if not answerable:
        if articles:
            raise ValueError(f"{qid}: an unanswerable question has no expected_articles")
    elif not articles:
        raise ValueError(f"{qid}: expected_articles is empty")
    if len(set(articles)) != len(articles):
        raise ValueError(f"{qid}: expected_articles has duplicates")
    if answerable and kind != "factual" and len(articles) < 2:
        raise ValueError(f"{qid}: a {kind} question needs at least 2 expected articles")
    equivalents_raw = data.get("equivalents") or {}
    if not isinstance(equivalents_raw, Mapping):
        raise ValueError(f"{qid}: equivalents must be an object")
    equivalents: dict[str, tuple[str, ...]] = {}
    for key, values in equivalents_raw.items():
        article = _ids(qid, [key], "equivalents")[0]
        if article not in articles:
            raise ValueError(f"{qid}: equivalents for {article}, which is not an expected article")
        equivalents[article] = _ids(qid, values, "equivalents")
    also_relevant = _ids(qid, data.get("also_relevant", []), "also_relevant")
    not_relevant = _ids(qid, data.get("judged_not_relevant", []), "judged_not_relevant")
    known = {*articles, *(i for ids in equivalents.values() for i in ids)}
    if known & {*also_relevant, *not_relevant}:
        raise ValueError(f"{qid}: an expected article cannot also be judged in also_relevant")
    if set(also_relevant) & set(not_relevant):
        raise ValueError(f"{qid}: an article cannot be both relevant and not relevant")
    if not answerable and also_relevant:
        raise ValueError(f"{qid}: an unanswerable question has no relevant article")
    validated = data.get("validated", False)
    if not isinstance(validated, bool):
        raise ValueError(f"{qid}: validated must be true or false")
    style = data.get("style", "named")
    if style not in QUESTION_STYLES:
        raise ValueError(f"{qid}: style must be one of {QUESTION_STYLES}, got {style!r}")
    return EvalQuestion(
        id=qid,
        type=kind,
        question=_text(data, "question"),
        expected_answer=_text(data, "expected_answer"),
        expected_articles=articles,
        equivalents=equivalents,
        validated=validated,
        style=style,
        also_relevant=also_relevant,
        judged_not_relevant=not_relevant,
        answerable=answerable,
    )


def check_unique_ids(questions: Sequence[EvalQuestion]) -> None:
    seen: set[str] = set()
    for question in questions:
        if question.id in seen:
            raise ValueError(f"duplicate question id {question.id}")
        seen.add(question.id)


def _text(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        ident = data.get("id", "?")
        raise ValueError(f"{ident}: {key} must be a non-empty text")
    return value.strip()


def _ids(qid: str, raw: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(raw, list | tuple):
        raise ValueError(f"{qid}: {field_name} must be a list of arXiv ids")
    try:
        return tuple(normalize_arxiv_id(str(item)) for item in raw)
    except ValueError as exc:
        raise ValueError(f"{qid}: {field_name}: {exc}") from None


def recall_at_k(groups: Sequence[frozenset[str]], retrieved: Sequence[str], k: int) -> float:
    """Share of expected articles found among the first k retrieved ids."""
    if not groups:
        raise ValueError("a question needs at least one expected article")
    top = set(retrieved[:k])
    return sum(1 for group in groups if group & top) / len(groups)


def reciprocal_rank(groups: Sequence[frozenset[str]], retrieved: Sequence[str]) -> float:
    """1 / rank of the first retrieved id that belongs to an expected article, else 0."""
    for rank, article in enumerate(retrieved, start=1):
        if any(article in group for group in groups):
            return 1.0 / rank
    return 0.0


EXPECTED_GRADE = 2  # an article the question was written from
ALSO_RELEVANT_GRADE = 1  # another article a person judged relevant


def ndcg_at_k(
    groups: Sequence[frozenset[str]],
    also_relevant: Collection[str],
    retrieved: Sequence[str],
    k: int,
) -> float:
    """Graded nDCG@k: expected articles weigh 2, `also_relevant` ones 1, the rest 0.

    Each expected article (with its equivalent ids) and each relevant article counts once, at
    its first appearance. The ideal ranking puts every expected article first, then the others.
    """
    if not groups:
        raise ValueError("a question needs at least one expected article")
    seen_groups: set[int] = set()
    seen_extra: set[str] = set()
    dcg = 0.0
    for rank, article in enumerate(retrieved[:k], start=1):
        grade = 0
        for index, group in enumerate(groups):
            if article in group and index not in seen_groups:
                seen_groups.add(index)
                grade = EXPECTED_GRADE
                break
        else:
            if article in also_relevant and article not in seen_extra:
                seen_extra.add(article)
                grade = ALSO_RELEVANT_GRADE
        dcg += grade / math.log2(rank + 1)
    ideal_grades = [EXPECTED_GRADE] * len(groups) + [ALSO_RELEVANT_GRADE] * len(set(also_relevant))
    ideal = sum(g / math.log2(r + 1) for r, g in enumerate(ideal_grades[:k], start=1))
    return dcg / ideal


@dataclass(frozen=True, slots=True)
class Summary:
    """Mean of per-question values with a 95% bootstrap interval."""

    mean: float
    low: float
    high: float
    n: int


def summarize(values: Sequence[float], *, resamples: int = 2000, seed: int = 0) -> Summary:
    """Mean and percentile bootstrap interval; the seed makes reports reproducible."""
    if not values:
        return Summary(0.0, 0.0, 0.0, 0)
    mean = statistics.fmean(values)
    if len(values) == 1:
        return Summary(mean, mean, mean, 1)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(resamples))
    low = means[int(0.025 * resamples)]
    high = means[min(int(0.975 * resamples), resamples - 1)]
    return Summary(mean, low, high, n)
