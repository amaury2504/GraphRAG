"""Evaluation questions and retrieval metrics (recall@k, MRR) with bootstrap intervals."""

import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from graphrag.domain.identifiers import normalize_arxiv_id

QuestionType = Literal["factual", "relational", "comparative", "evolutive"]
QUESTION_TYPES: tuple[QuestionType, ...] = get_args(QuestionType)


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    """A question with the articles that answer it.

    `equivalents` maps an expected article to other arXiv ids of the same work (a paper
    deposited twice): finding any of them counts as finding the article.
    """

    id: str
    type: QuestionType
    question: str
    expected_answer: str
    expected_articles: tuple[str, ...]
    equivalents: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    validated: bool = False

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
    articles = _ids(qid, data.get("expected_articles"), "expected_articles")
    if not articles:
        raise ValueError(f"{qid}: expected_articles is empty")
    if len(set(articles)) != len(articles):
        raise ValueError(f"{qid}: expected_articles has duplicates")
    if kind != "factual" and len(articles) < 2:
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
    validated = data.get("validated", False)
    if not isinstance(validated, bool):
        raise ValueError(f"{qid}: validated must be true or false")
    return EvalQuestion(
        id=qid,
        type=kind,
        question=_text(data, "question"),
        expected_answer=_text(data, "expected_answer"),
        expected_articles=articles,
        equivalents=equivalents,
        validated=validated,
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
