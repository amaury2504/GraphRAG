"""Pool the unjudged results of the retrievers for a person to judge, then record the verdicts.

    python -m graphrag.cli.pool                      write eval/pool_review.jsonl
    python -m graphrag.cli.pool --apply FILE          copy the verdicts into eval/questions.jsonl

In the review file, set `"relevant": true` for an article that answers the question as well,
`false` for one that does not; leave `null` to decide later. Calls no language model.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from graphrag.adapters.neo4j_store import Neo4jStore
from graphrag.adapters.question_file import DEFAULT_PATH, read_questions, write_records
from graphrag.adapters.sentence_transformer import default_index_info, load_default_embedder
from graphrag.domain.evaluation import parse_question
from graphrag.pipeline.pooling import POOL_DEPTH, apply_judgements, build_pool, review_record
from graphrag.pipeline.retrieval import HybridRetriever, VectorRetriever
from graphrag.ports.retrieval import Retriever
from graphrag.settings import Settings

REVIEW_PATH = Path("eval/pool_review.jsonl")


def read_lines(path: Path) -> list[dict[str, Any]]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON ({exc.msg})") from None
        if not isinstance(data, dict):
            raise ValueError(f"{path}:{number}: a line must be a JSON object")
        records.append(data)
    return records


def apply(review: Path, questions_path: Path) -> int:
    """Record the verdicts of a review file; nothing is written if anything is inconsistent."""
    try:
        updated, relevant, not_relevant, pending = apply_judgements(
            read_lines(questions_path), read_lines(review)
        )
        for record in updated:
            parse_question(record)
    except ValueError as exc:
        print(f"nothing written: {exc}")
        return 2
    write_records(questions_path, updated)
    print(
        f"{relevant} relevant, {not_relevant} not relevant recorded in {questions_path}; "
        f"{pending} still to judge in {review}"
    )
    return 0


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pool unjudged results for judging.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--output", type=Path, default=REVIEW_PATH)
    parser.add_argument("--depth", type=int, default=POOL_DEPTH)
    parser.add_argument("--force", action="store_true", help="overwrite the review file")
    parser.add_argument("--apply", type=Path, metavar="REVIEW", help="record judged verdicts")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.apply is not None:
        return apply(args.apply, args.questions)
    if args.output.exists() and not args.force:
        print(f"{args.output} already exists (judge it and --apply it, or pass --force)")
        return 2
    questions = [q for q in read_questions(args.questions) if q.validated]
    if not questions:
        print("No validated question to pool.")
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
        items = build_pool(questions, retrievers, args.depth)
    write_records(args.output, [review_record(i) for i in items])
    print(f"{len(items)} articles to judge for {len(questions)} questions: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
