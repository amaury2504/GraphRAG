"""Pooling: collect the articles the retrievers return that nobody has judged yet.

The corpus only talks about RAG, so an article other than the expected one may answer a
question too. Counted as a miss, it would punish a system for a good result. Each retriever's
top results that are neither expected nor already judged are shown to a person, whose answers
are stored in the question (`also_relevant`, `judged_not_relevant`). Pool again whenever a new
configuration (e.g. `graph`) is added: its new results have not been judged either.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from graphrag.domain.evaluation import EvalQuestion
from graphrag.domain.identifiers import normalize_arxiv_id
from graphrag.domain.models import SearchHit
from graphrag.ports.retrieval import Retriever

POOL_DEPTH = 10


@dataclass(frozen=True, slots=True)
class PoolItem:
    question_id: str
    question: str
    arxiv_id: str
    title: str
    abstract: str
    found_by: Mapping[str, int]  # configuration -> rank (from 1)


def judged_or_expected(question: EvalQuestion) -> set[str]:
    """Ids that need no judgement: expected (with equivalents) or already judged."""
    expected = {article for group in question.groups for article in group}
    return expected | set(question.also_relevant) | set(question.judged_not_relevant)


def build_pool(
    questions: Sequence[EvalQuestion],
    retrievers: Mapping[str, Retriever],
    depth: int = POOL_DEPTH,
) -> list[PoolItem]:
    """The unjudged articles among each retriever's top `depth`: one item per question and article.

    Unanswerable questions have nothing to judge and are skipped.
    """
    items: list[PoolItem] = []
    for question in questions:
        if not question.answerable:
            continue
        skip = judged_or_expected(question)
        ranks: dict[str, dict[str, int]] = {}
        first_hit: dict[str, SearchHit] = {}
        for name, retriever in retrievers.items():
            for rank, hit in enumerate(retriever.retrieve(question.question, depth), start=1):
                if hit.arxiv_id in skip:
                    continue
                ranks.setdefault(hit.arxiv_id, {})[name] = rank
                first_hit.setdefault(hit.arxiv_id, hit)
        for arxiv_id, found_by in ranks.items():
            hit = first_hit[arxiv_id]
            items.append(
                PoolItem(
                    question.id,
                    question.question,
                    arxiv_id,
                    hit.title,
                    hit.abstract or hit.text,
                    found_by,
                )
            )
    return items


def review_record(item: PoolItem) -> dict[str, Any]:
    """One line of the review file: fill `relevant` with true or false."""
    return {
        "question_id": item.question_id,
        "question": item.question,
        "arxiv_id": item.arxiv_id,
        "title": item.title,
        "abstract": item.abstract,
        "found_by": dict(item.found_by),
        "relevant": None,
    }


def apply_judgements(
    records: Sequence[dict[str, Any]], judgements: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Copy the judgements into the question records.

    Returns the updated records and the counts (relevant, not relevant, not judged yet).
    `relevant: null` is left for later; anything but true, false or null is an error, as is an
    unknown question, an article already expected for that question, or one judged both ways.
    """
    by_id = {str(r["id"]): r for r in records}
    relevant = not_relevant = pending = 0
    for number, judgement in enumerate(judgements, start=1):
        qid = str(judgement.get("question_id"))
        if qid not in by_id:
            raise ValueError(f"judgement {number}: unknown question {qid!r}")
        verdict = judgement.get("relevant")
        if verdict is None:
            pending += 1
            continue
        if not isinstance(verdict, bool):
            raise ValueError(f"judgement {number} ({qid}): relevant must be true, false or null")
        article = normalize_arxiv_id(str(judgement.get("arxiv_id")))
        record = by_id[qid]
        expected = {normalize_arxiv_id(a) for a in record.get("expected_articles", [])}
        if article in expected:
            raise ValueError(f"judgement {number} ({qid}): {article} is an expected article")
        target, other = (
            ("also_relevant", "judged_not_relevant")
            if verdict
            else ("judged_not_relevant", "also_relevant")
        )
        if article in record.get(other, []):
            raise ValueError(f"judgement {number} ({qid}): {article} was judged the other way")
        if article not in record.setdefault(target, []):
            record[target].append(article)
        if verdict:
            relevant += 1
        else:
            not_relevant += 1
    return list(by_id.values()), relevant, not_relevant, pending
