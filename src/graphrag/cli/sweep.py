"""Sweep the chunk size and overlap: index the corpus with each setting, evaluate retrieval.

The articles are read from the real database (read only) and copied into a throwaway Neo4j
(`docker compose --profile test up -d --wait neo4j-test`, port 7688), where every configuration
is indexed in turn: the real database is never indexed with an experimental chunker. Calls no
language model; the embedding model runs locally.

Choose on the `tune` half, then check the winner once on the `report` half (see `evaluate`).
Changing the default settings afterwards means editing `chunking.py`, raising CHUNKER_VERSION
and re-indexing the real database: not done here.
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from graphrag.adapters.neo4j_store import Neo4jStore
from graphrag.adapters.question_file import DEFAULT_PATH, read_questions
from graphrag.adapters.sentence_transformer import index_info_for, load_embedder
from graphrag.cli.evaluate import select_questions
from graphrag.cli.index import run_indexing
from graphrag.cli.ingest import log_line
from graphrag.domain.chunking import ChunkerConfig
from graphrag.domain.evaluation import SPLITS, EvalQuestion
from graphrag.domain.models import Article, IndexInfo
from graphrag.pipeline.evaluation import DEFAULT_KS, NDCG_K, QuestionResult, summarize_results
from graphrag.pipeline.retrieval import HybridRetriever, VectorRetriever
from graphrag.pipeline.sweep import (
    BASELINE,
    OVERLAPS,
    SIZES,
    SweepPoint,
    default_grid,
    label,
    run_sweep,
    sweep_table,
)
from graphrag.ports.embedding import Embedder
from graphrag.ports.retrieval import Retriever
from graphrag.settings import Settings

RESULTS_DIR = Path("eval/results")
DEFAULT_TARGET_URI = "bolt://localhost:7688"
MIN_QUESTIONS_NOTE = 30


class SameDatabaseError(ValueError):
    """The sweep target is the database the articles are read from."""


def check_target(source_uri: str, target_uri: str) -> None:
    if source_uri.rstrip("/") == target_uri.rstrip("/"):
        raise SameDatabaseError(
            f"{target_uri} is the real database: a sweep indexes experimental chunks, so it "
            "needs the throwaway one (docker compose --profile test up -d --wait neo4j-test, "
            f"then --target-uri {DEFAULT_TARGET_URI})"
        )


def grid_from(sizes: Sequence[int] | None, overlaps: Sequence[int] | None) -> list[ChunkerConfig]:
    """The default grid, or every size and overlap asked for (baseline always first)."""
    if sizes is None and overlaps is None:
        return default_grid()
    grid = [BASELINE]
    for size in sizes or SIZES:
        for overlap in overlaps if overlaps is not None else OVERLAPS:
            config = ChunkerConfig(size, overlap)
            if config not in grid:
                grid.append(config)
    return grid


def sweep_corpus(
    articles: Sequence[Article],
    store: Neo4jStore,
    embedder: Embedder,
    info: IndexInfo,
    questions: Sequence[EvalQuestion],
    configs: Sequence[ChunkerConfig],
    ks: Sequence[int],
    log: Callable[[str], None] = log_line,
) -> list[SweepPoint]:
    """Copy `articles` into `store` (a throwaway database), then sweep `configs` over it."""
    store.ensure_schema()
    store.upsert_articles(articles)

    def index(config: ChunkerConfig) -> int:
        log(f"sweep: indexing with {label(config)}")
        run_indexing(store, embedder, info=info, config=config, log=log)
        version = store.current_version()
        if version is None:
            raise RuntimeError("no version published after indexing")
        return version.chunk_count

    retrievers: dict[str, Retriever] = {
        "vector": VectorRetriever(store, embedder),
        "hybrid": HybridRetriever(store, embedder),
    }
    return run_sweep(configs, questions, retrievers, ks, index)


def build_report(
    points: Sequence[SweepPoint],
    questions: Sequence[EvalQuestion],
    ks: Sequence[int],
    *,
    split: str | None,
    include_unvalidated: bool,
    articles: int,
    embedding: tuple[str, str],
    now: datetime,
) -> dict[str, Any]:

    def figures(results: Sequence[QuestionResult]) -> dict[str, Any]:
        overall = summarize_results(results, ks)["all"]
        return {
            "mrr": overall.mrr.mean,
            f"ndcg@{NDCG_K}": overall.ndcg.mean,
            "recall": {str(k): overall.recall[k].mean for k in ks},
        }

    return {
        "created": now.isoformat(timespec="seconds"),
        "kind": "chunk_sweep",
        "provisional": include_unvalidated,
        "split": split or "all",
        "questions": len(questions),
        "articles": articles,
        "ks": list(ks),
        "embedding_model": embedding[0],
        "embedding_revision": embedding[1],
        "points": [
            {
                "config": asdict(p.config),
                "label": label(p.config),
                "chunks": p.chunks,
                "summary": {name: figures(results) for name, results in p.results.items()},
                "questions": {name: [asdict(r) for r in rs] for name, rs in p.results.items()},
            }
            for p in points
        ],
    }


def render(
    points: Sequence[SweepPoint],
    questions: Sequence[EvalQuestion],
    ks: Sequence[int],
    *,
    include_unvalidated: bool,
    split: str | None,
) -> str:
    parts = []
    if include_unvalidated:
        parts.append("PROVISIONAL: unvalidated questions are included.")
    if split == "tune":
        parts.append("TUNING HALF: choose settings here, never quote these figures as results.")
    elif split == "report":
        parts.append("REPORT HALF: do not change any setting after seeing these figures.")
    parts.append(f"{len(questions)} questions")
    parts.append(sweep_table(points, ks))
    parts.append(
        "* = the 95% interval of the MRR gap excludes 0. Few questions give wide intervals: "
        "a gap that does not exclude 0 is not a result."
    )
    if len(questions) < MIN_QUESTIONS_NOTE:
        parts.append(f"Only {len(questions)} questions: do not pick a setting from this sweep.")
    return "\n\n".join(parts)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep chunk size and overlap (recall@k, MRR).")
    parser.add_argument("--questions", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--include-unvalidated", action="store_true")
    parser.add_argument("--split", choices=SPLITS, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=None, help="max words per chunk")
    parser.add_argument(
        "--overlaps", type=int, nargs="+", default=None, help="sentences repeated between chunks"
    )
    parser.add_argument(
        "--target-uri", default=DEFAULT_TARGET_URI, help="the throwaway Neo4j the sweep indexes in"
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    questions = select_questions(
        read_questions(args.questions), args.include_unvalidated, args.split
    )
    if not questions:
        print("No validated question to evaluate (use --include-unvalidated for a trial run).")
        return 1
    settings = Settings()  # type: ignore[call-arg]  # values come from the environment
    try:
        check_target(settings.neo4j_uri, args.target_uri)
    except SameDatabaseError as error:
        print(error)
        return 2
    user, password = settings.neo4j_user, settings.neo4j_password.get_secret_value()
    with Neo4jStore(settings.neo4j_uri, user, password) as source:
        articles = source.all_articles()
    log_line("loading the embedding model")
    embedder = load_embedder(settings.embedding_model)
    info = index_info_for(settings.embedding_model, embedder.dimensions)
    ks = sorted(set(args.ks))
    configs = grid_from(args.sizes, args.overlaps)
    with Neo4jStore(args.target_uri, user, password) as target:
        points = sweep_corpus(articles, target, embedder, info, questions, configs, ks)
    print(
        render(
            points, questions, ks, include_unvalidated=args.include_unvalidated, split=args.split
        )
    )
    now = datetime.now(UTC)
    report = build_report(
        points,
        questions,
        ks,
        split=args.split,
        include_unvalidated=args.include_unvalidated,
        articles=len(articles),
        embedding=(info.model, info.revision),
        now=now,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"sweep_{now:%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
