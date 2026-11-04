"""Retrieval evaluation: recall@k and MRR per configuration, overall and by question type.

Calls no language model. Questions come from eval/questions.jsonl; only validated ones are
used unless --include-unvalidated is given (the report is then marked provisional).
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from graphrag.adapters.neo4j_store import Neo4jStore
from graphrag.adapters.question_file import DEFAULT_PATH, read_questions
from graphrag.adapters.sentence_transformer import (
    MODEL_NAME,
    MODEL_REVISION,
    default_index_info,
    load_default_embedder,
)
from graphrag.domain.evaluation import EvalQuestion, Summary
from graphrag.pipeline.evaluation import (
    DEFAULT_KS,
    NDCG_K,
    GroupSummary,
    QuestionResult,
    evaluate_retrieval,
    paired_differences,
    summarize_results,
)
from graphrag.pipeline.retrieval import (
    CANDIDATES,
    RRF_K,
    HybridRetriever,
    KeywordRetriever,
    VectorRetriever,
)
from graphrag.ports.retrieval import Retriever
from graphrag.settings import Settings

RESULTS_DIR = Path("eval/results")
MIN_QUESTIONS_NOTE = 30
MIN_QUESTIONS_FOR_STAR = 10


def select_questions(
    questions: Sequence[EvalQuestion], include_unvalidated: bool
) -> list[EvalQuestion]:
    return [q for q in questions if include_unvalidated or q.validated]


def format_table(name: str, summaries: Mapping[str, GroupSummary], ks: Sequence[int]) -> str:
    header = (
        f"{'group':<12}{'n':>4}"
        + "".join(f"{f'R@{k}':>7}" for k in ks)
        + f"{'MRR':>7}{f'nDCG@{NDCG_K}':>9}{'s/q':>7}"
    )
    lines = [f"== {name} ==", header]
    for group, summary in summaries.items():
        row = f"{group:<12}{summary.n:>4}"
        row += "".join(f"{summary.recall[k].mean:>7.2f}" for k in ks)
        row += f"{summary.mrr.mean:>7.2f}{summary.ndcg.mean:>9.2f}{summary.mean_seconds:>7.3f}"
        lines.append(row)
    return "\n".join(lines)


def _interval(summary: Summary, n: int) -> str:
    excludes_zero = summary.low > 0 or summary.high < 0
    mark = "*" if excludes_zero and n >= MIN_QUESTIONS_FOR_STAR else " "
    return f"{summary.mean:+.2f} [{summary.low:+.2f}, {summary.high:+.2f}]{mark}"


def format_differences(
    baseline: str,
    other: str,
    differences: Mapping[str, Mapping[str, Summary]],
    sizes: Mapping[str, int],
    ks: Sequence[int],
) -> str:
    focus = f"recall@{5 if 5 in ks else max(ks)}"
    lines = [
        f"== {other} minus {baseline}: paired, 95% bootstrap interval over questions ==",
        f"{'group':<12}{'n':>4}  {'MRR':<26}{focus:<26}{f'ndcg@{NDCG_K}':<26}",
    ]
    for group, metrics in differences.items():
        n = sizes[group]
        mrr_cell = _interval(metrics["mrr"], n)
        focus_cell = _interval(metrics[focus], n)
        ndcg_cell = _interval(metrics[f"ndcg@{NDCG_K}"], n)
        lines.append(f"{group:<12}{n:>4}  {mrr_cell:<26}{focus_cell:<26}{ndcg_cell:<26}")
    lines.append(
        "* = the interval excludes 0; otherwise the gap may be chance. "
        f"Never starred under {MIN_QUESTIONS_FOR_STAR} questions: the interval is then too narrow."
    )
    return "\n".join(lines)


def _summary_json(summary: Summary) -> dict[str, float | int]:
    return {"mean": summary.mean, "low": summary.low, "high": summary.high, "n": summary.n}


def build_report(
    questions: Sequence[EvalQuestion],
    results: Mapping[str, Sequence[QuestionResult]],
    ks: Sequence[int],
    *,
    include_unvalidated: bool,
    now: datetime,
) -> dict[str, Any]:
    names = list(results)
    configs: dict[str, Any] = {}
    for name in names:
        summaries = summarize_results(results[name], ks)
        configs[name] = {
            "summary": {
                group: {
                    "n": s.n,
                    "recall": {str(k): _summary_json(s.recall[k]) for k in ks},
                    "mrr": _summary_json(s.mrr),
                    f"ndcg@{NDCG_K}": _summary_json(s.ndcg),
                    "mean_seconds": s.mean_seconds,
                }
                for group, s in summaries.items()
            },
            "questions": [asdict(r) for r in results[name]],
        }
    differences: dict[str, Any] = {}
    for other in names[1:]:
        raw = paired_differences(results[names[0]], results[other], ks)
        differences[f"{other} minus {names[0]}"] = {
            group: {metric: _summary_json(s) for metric, s in metrics.items()}
            for group, metrics in raw.items()
        }
    return {
        "created": now.isoformat(timespec="seconds"),
        "provisional": include_unvalidated,
        "questions": len(questions),
        "ks": list(ks),
        "settings": {
            "embedding_model": MODEL_NAME,
            "embedding_revision": MODEL_REVISION,
            "rrf_k": RRF_K,
            "candidates_per_search": CANDIDATES,
        },
        "configs": configs,
        "differences": differences,
    }


def render(
    questions: Sequence[EvalQuestion],
    results: Mapping[str, Sequence[QuestionResult]],
    ks: Sequence[int],
    *,
    include_unvalidated: bool,
) -> str:
    names = list(results)
    parts = []
    if include_unvalidated:
        parts.append("PROVISIONAL: unvalidated questions are included.")
    parts.append(f"{len(questions)} questions")
    for name in names:
        parts.append(format_table(name, summarize_results(results[name], ks), ks))
    for other in names[1:]:
        diffs = paired_differences(results[names[0]], results[other], ks)
        sizes = {g: s.n for g, s in summarize_results(results[names[0]], ks).items()}
        parts.append(format_differences(names[0], other, diffs, sizes, ks))
    if len(questions) < MIN_QUESTIONS_NOTE:
        parts.append(
            f"Only {len(questions)} questions: per-type figures rest on very few questions, "
            "and small gaps are not significant."
        )
    return "\n\n".join(parts)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval (recall@k, MRR).")
    parser.add_argument("--questions", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--include-unvalidated", action="store_true")
    parser.add_argument(
        "--with-keyword",
        action="store_true",
        help="also evaluate keyword-only search (diagnostic: what each half brings to the fusion)",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    questions = select_questions(read_questions(args.questions), args.include_unvalidated)
    if not questions:
        print("No validated question to evaluate (use --include-unvalidated for a trial run).")
        return 1
    settings = Settings()  # type: ignore[call-arg]  # values come from the environment
    embedder = load_default_embedder()
    with Neo4jStore(
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password.get_secret_value()
    ) as store:
        store.check_index_info(default_index_info(embedder.dimensions))
        retrievers: dict[str, Retriever] = {
            "vector": VectorRetriever(store, embedder),
            "hybrid": HybridRetriever(store, embedder),
        }
        if args.with_keyword:  # diagnostic only: appended last, so the baseline stays "vector"
            retrievers["keyword"] = KeywordRetriever(store)
        ks = sorted(set(args.ks))
        results = evaluate_retrieval(questions, retrievers, ks)
    print(render(questions, results, ks, include_unvalidated=args.include_unvalidated))
    now = datetime.now(UTC)
    report = build_report(
        questions, results, ks, include_unvalidated=args.include_unvalidated, now=now
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"retrieval_{now:%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
