"""Run the retrievers on the evaluation questions and compute recall@k and MRR.

No language model is involved: this part is free and deterministic.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from graphrag.domain.evaluation import (
    QUESTION_TYPES,
    EvalQuestion,
    Summary,
    recall_at_k,
    reciprocal_rank,
    summarize,
)
from graphrag.ports.retrieval import Retriever

DEFAULT_KS = (1, 3, 5, 10)
ALL = "all"


@dataclass(frozen=True, slots=True)
class QuestionResult:
    question_id: str
    type: str
    retrieved: tuple[str, ...]  # arXiv ids, best first
    recall: Mapping[int, float]  # k -> recall@k
    reciprocal_rank: float
    seconds: float


@dataclass(frozen=True, slots=True)
class GroupSummary:
    n: int
    recall: Mapping[int, Summary]
    mrr: Summary
    mean_seconds: float


def evaluate_retrieval(
    questions: Sequence[EvalQuestion],
    retrievers: Mapping[str, Retriever],
    ks: Sequence[int] = DEFAULT_KS,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, list[QuestionResult]]:
    """Ask every question to every retriever, fetching max(ks) articles each time."""
    if not ks or min(ks) < 1:
        raise ValueError("ks must be positive integers")
    depth = max(ks)
    results: dict[str, list[QuestionResult]] = {}
    for name, retriever in retrievers.items():
        per_question = []
        for question in questions:
            start = clock()
            hits = retriever.retrieve(question.question, depth)
            seconds = clock() - start
            retrieved = tuple(hit.arxiv_id for hit in hits)
            groups = question.groups
            per_question.append(
                QuestionResult(
                    question_id=question.id,
                    type=question.type,
                    retrieved=retrieved,
                    recall={k: recall_at_k(groups, retrieved, k) for k in ks},
                    reciprocal_rank=reciprocal_rank(groups, retrieved),
                    seconds=seconds,
                )
            )
        results[name] = per_question
    return results


def _group(results: Sequence[QuestionResult], ks: Sequence[int]) -> GroupSummary:
    return GroupSummary(
        n=len(results),
        recall={k: summarize([r.recall[k] for r in results]) for k in ks},
        mrr=summarize([r.reciprocal_rank for r in results]),
        mean_seconds=sum(r.seconds for r in results) / len(results) if results else 0.0,
    )


def summarize_results(
    results: Sequence[QuestionResult], ks: Sequence[int] = DEFAULT_KS
) -> dict[str, GroupSummary]:
    """Overall summary under "all", then one per question type that has questions."""
    summaries = {ALL: _group(results, ks)}
    for kind in QUESTION_TYPES:
        of_type = [r for r in results if r.type == kind]
        if of_type:
            summaries[kind] = _group(of_type, ks)
    return summaries


def paired_differences(
    baseline: Sequence[QuestionResult],
    other: Sequence[QuestionResult],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, dict[str, Summary]]:
    """Per-question difference `other - baseline` for MRR and each recall@k, by group.

    The interval is a bootstrap over questions: when it contains 0, the gap between the two
    configurations is not distinguishable from chance on this question set.
    """
    if [r.question_id for r in baseline] != [r.question_id for r in other]:
        raise ValueError("both configurations must be evaluated on the same questions")
    groups: dict[str, list[tuple[QuestionResult, QuestionResult]]] = {
        ALL: list(zip(baseline, other, strict=True))
    }
    for kind in QUESTION_TYPES:
        pairs = [(a, b) for a, b in zip(baseline, other, strict=True) if a.type == kind]
        if pairs:
            groups[kind] = pairs
    differences: dict[str, dict[str, Summary]] = {}
    for name, pairs in groups.items():
        metrics = {"mrr": summarize([b.reciprocal_rank - a.reciprocal_rank for a, b in pairs])}
        for k in ks:
            metrics[f"recall@{k}"] = summarize([b.recall[k] - a.recall[k] for a, b in pairs])
        differences[name] = metrics
    return differences
