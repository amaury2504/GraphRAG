"""Sweep the chunker settings: index the corpus with each one, evaluate retrieval, compare.

The first configuration is the baseline; the others are compared to it with the paired
differences of `evaluation`. Indexing is a callback so that this stays pure: the CLI gives it a
store (the throwaway database) and an embedder.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from graphrag.domain.chunking import ChunkerConfig
from graphrag.domain.evaluation import EvalQuestion
from graphrag.pipeline.evaluation import (
    ALL,
    NDCG_K,
    QuestionResult,
    evaluate_retrieval,
    paired_differences,
    summarize_results,
)
from graphrag.ports.retrieval import Retriever

BASELINE = ChunkerConfig()
SIZES = (60, 100, 150, 200)
OVERLAPS = (0, 1)


@dataclass(frozen=True, slots=True)
class SweepPoint:
    config: ChunkerConfig
    chunks: int  # chunks in the version read by the searches
    results: Mapping[str, list[QuestionResult]]  # by retriever name


def label(config: ChunkerConfig) -> str:
    return f"w{config.max_words}/o{config.overlap_sentences}"


def default_grid() -> list[ChunkerConfig]:
    """The baseline first, then every other size and overlap pair."""
    grid = [BASELINE]
    grid += [
        ChunkerConfig(size, overlap)
        for size in SIZES
        for overlap in OVERLAPS
        if ChunkerConfig(size, overlap) != BASELINE
    ]
    return grid


def run_sweep(
    configs: Sequence[ChunkerConfig],
    questions: Sequence[EvalQuestion],
    retrievers: Mapping[str, Retriever],
    ks: Sequence[int],
    index: Callable[[ChunkerConfig], int],
) -> list[SweepPoint]:
    """`index(config)` makes the corpus searchable chunked that way and returns its chunk count."""
    if not configs:
        raise ValueError("a sweep needs at least the baseline configuration")
    if configs[0] != BASELINE:
        raise ValueError(f"the baseline ({label(BASELINE)}) must come first")
    points = []
    for config in configs:
        chunks = index(config)
        points.append(SweepPoint(config, chunks, evaluate_retrieval(questions, retrievers, ks)))
    return points


def sweep_table(points: Sequence[SweepPoint], ks: Sequence[int]) -> str:
    """One block per retriever: figures per configuration, and the MRR gap to the baseline."""
    focus = 5 if 5 in ks else max(ks)
    names = list(points[0].results)
    blocks = []
    for name in names:
        header = (
            f"{'config':<10}{'chunks':>8}{f'R@{focus}':>7}{'MRR':>7}{f'nDCG@{NDCG_K}':>9}"
            f"{'dMRR vs baseline':>20}"
        )
        lines = [f"== {name} (all questions) ==", header]
        baseline = points[0].results[name]
        for point in points:
            summary = summarize_results(point.results[name], ks)[ALL]
            if point is points[0]:
                gap = "baseline"
            else:
                diff = paired_differences(baseline, point.results[name], ks)[ALL]["mrr"]
                mark = "*" if diff.low > 0 or diff.high < 0 else ""
                gap = f"{diff.mean:+.2f} [{diff.low:+.2f}, {diff.high:+.2f}]{mark}"
            lines.append(
                f"{label(point.config):<10}{point.chunks:>8}{summary.recall[focus].mean:>7.2f}"
                f"{summary.mrr.mean:>7.2f}{summary.ndcg.mean:>9.2f}{gap:>20}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
