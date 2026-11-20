import pytest

from graphrag.domain.chunking import ChunkerConfig
from graphrag.domain.evaluation import EvalQuestion
from graphrag.domain.models import SearchHit
from graphrag.pipeline.sweep import (
    BASELINE,
    SweepPoint,
    default_grid,
    label,
    run_sweep,
    sweep_table,
)

KS = (1, 5)


def q(qid: str, *articles: str) -> EvalQuestion:
    return EvalQuestion(qid, "factual", f"{qid}?", "a", articles, {}, True)


class Scripted:
    """Answers depend on which configuration is currently 'indexed'."""

    def __init__(self, state: dict[str, ChunkerConfig], good: ChunkerConfig) -> None:
        self.state = state
        self.good = good

    def retrieve(self, question: str, k: int) -> list[SearchHit]:
        ids = ["A"] if self.state["current"] == self.good else ["x", "A"]
        return [SearchHit(i, "t", 0, "x", 0.0) for i in ids][:k]


QUESTIONS = [q("f1", "A"), q("f2", "A")]


def sweep(configs: list[ChunkerConfig], good: ChunkerConfig) -> list[SweepPoint]:
    state = {"current": configs[0]}
    indexed: list[ChunkerConfig] = []

    def index(config: ChunkerConfig) -> int:
        state["current"] = config
        indexed.append(config)
        return config.max_words

    retrievers = {"vector": Scripted(state, good), "hybrid": Scripted(state, good)}
    points = run_sweep(configs, QUESTIONS, retrievers, KS, index)
    assert indexed == configs
    return points


def test_default_grid_starts_with_the_baseline_and_covers_sizes_and_overlaps() -> None:
    grid = default_grid()
    assert grid[0] == BASELINE == ChunkerConfig()
    assert len(grid) == len(set(grid)) == 8
    assert {c.max_words for c in grid} == {60, 100, 150, 200}
    assert {c.overlap_sentences for c in grid} == {0, 1}


def test_label_names_size_and_overlap() -> None:
    assert label(ChunkerConfig(100, 0)) == "w100/o0"


def test_every_configuration_is_indexed_then_evaluated_in_order() -> None:
    configs = [BASELINE, ChunkerConfig(100, 1), ChunkerConfig(100, 0)]
    points = sweep(configs, good=ChunkerConfig(100, 1))
    assert [p.config for p in points] == configs
    assert [p.chunks for p in points] == [200, 100, 100]
    assert [r.reciprocal_rank for r in points[1].results["vector"]] == [1.0, 1.0]
    assert [r.reciprocal_rank for r in points[0].results["vector"]] == [0.5, 0.5]


def test_the_baseline_must_come_first() -> None:
    with pytest.raises(ValueError, match="baseline"):
        sweep([ChunkerConfig(100, 1), BASELINE], good=BASELINE)


def test_an_empty_grid_is_refused() -> None:
    with pytest.raises(ValueError, match="at least"):
        run_sweep([], QUESTIONS, {}, KS, lambda _c: 0)


def test_table_shows_each_configuration_and_the_gap_to_the_baseline() -> None:
    configs = [BASELINE, ChunkerConfig(100, 1)]
    table = sweep_table(sweep(configs, good=ChunkerConfig(100, 1)), KS)
    lines = table.splitlines()
    assert "w200/o1" in table
    assert "w100/o1" in table
    assert any("w100/o1" in line and "+0.50" in line for line in lines)  # MRR 0.5 -> 1.0
    assert "baseline" in table
