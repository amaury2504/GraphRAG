"""Write candidate evaluation questions with the LLM (costs tokens; small batches only).

Articles are chosen by topic proximity (see pipeline.question_generation). The candidates are
written to eval/candidates.jsonl with "validated": false: read each one, fix it, then move it
to eval/questions.jsonl with "validated": true.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from graphrag.adapters.mistral import MistralLLM, mistral_client
from graphrag.adapters.neo4j_store import Neo4jStore
from graphrag.adapters.question_file import (
    CANDIDATES_PATH,
    DEFAULT_PATH,
    read_questions,
    write_records,
)
from graphrag.adapters.sentence_transformer import load_default_embedder
from graphrag.cli.ingest import log_line
from graphrag.domain.evaluation import QUESTION_TYPES, EvalQuestion
from graphrag.pipeline.question_generation import (
    generate_candidates,
    plan_questions,
    title_groups,
)
from graphrag.pipeline.usage import MeteredLLM, UsageMeter
from graphrag.settings import Settings

MAX_WITHOUT_CONFIRMATION = 8  # questions: the user fixes the size of the first batch


def used_articles(questions: Sequence[EvalQuestion]) -> set[str]:
    """Every article (and equivalent id) that existing questions already ask about."""
    used: set[str] = set()
    for question in questions:
        for group in question.groups:
            used.update(group)
    return used


def check_request(
    per_type: int, type_count: int, confirm_cost: bool, output: Path, force: bool
) -> str | None:
    """Return the reason this run must not start, or None."""
    if per_type < 1:
        return "--per-type must be at least 1"
    total = per_type * type_count
    if total > MAX_WITHOUT_CONFIRMATION and not confirm_cost:
        return (
            f"{total} questions means {total} LLM calls; more than {MAX_WITHOUT_CONFIRMATION} "
            "needs --confirm-cost once you have agreed to the cost"
        )
    if output.exists() and not force:
        return f"{output} already exists; move it away or pass --force to overwrite it"
    return None


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate candidate evaluation questions.")
    parser.add_argument("--per-type", type=int, default=2, help="questions of each of the 4 types")
    parser.add_argument("--seed", type=int, default=42, help="makes the choice of articles repeat")
    parser.add_argument(
        "--first-index", type=int, default=1, help="number of the first id per type"
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=QUESTION_TYPES,
        default=list(QUESTION_TYPES),
        help="question types to generate (default: all four)",
    )
    parser.add_argument("--output", type=Path, default=CANDIDATES_PATH)
    parser.add_argument(
        "--exclude-questions",
        type=Path,
        default=DEFAULT_PATH,
        help="never reuse the articles of these questions (skipped if the file is missing)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite the output file")
    parser.add_argument("--confirm-cost", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="choose the articles, call no LLM")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    problem = check_request(
        args.per_type, len(args.types), args.confirm_cost, args.output, args.force
    )
    if problem:
        print(problem)
        return 2
    settings = Settings()  # type: ignore[call-arg]  # values come from the environment
    key = settings.mistral_api_key.get_secret_value() if settings.mistral_api_key else ""
    if not key and not args.dry_run:
        print("MISTRAL_API_KEY is not set")
        return 2
    embedder = load_default_embedder()
    meter = UsageMeter(settings.token_alert)
    with Neo4jStore(
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password.get_secret_value()
    ) as store:
        articles = store.all_articles()
        existing = read_questions(args.exclude_questions) if args.exclude_questions.exists() else []
        exclude = used_articles(existing)
        log_line(f"{len(exclude)} articles excluded: already used by {len(existing)} questions")
        plans = plan_questions(
            articles,
            store,
            embedder,
            per_type=args.per_type,
            seed=args.seed,
            types=args.types,
            exclude=exclude,
        )
    for plan in plans:
        log_line(
            f"{plan.type}: " + " | ".join(f"{a.arxiv_id} {a.title[:50]}" for a in plan.articles)
        )
    if args.dry_run:
        print(f"dry run: {len(plans)} groups chosen, no LLM call, nothing written")
        return 0
    llm = MeteredLLM(MistralLLM(mistral_client(key), model=settings.mistral_model), meter)
    records, failures = generate_candidates(
        plans, llm, title_groups(articles), first_index=args.first_index, log=log_line
    )
    write_records(args.output, records)
    print(
        f"{len(records)} candidates written to {args.output}, {len(failures)} failed; "
        f"{meter.total.total_tokens} tokens in {meter.calls} calls"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
