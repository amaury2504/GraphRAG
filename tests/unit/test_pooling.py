import json
from pathlib import Path
from typing import Any

import pytest

from graphrag.cli.pool import apply, read_lines
from graphrag.domain.evaluation import EvalQuestion, parse_question
from graphrag.domain.models import SearchHit
from graphrag.pipeline.pooling import apply_judgements, build_pool, review_record


def question(
    qid: str = "q1",
    articles: tuple[str, ...] = ("2401.00001",),
    **fields: Any,
) -> EvalQuestion:
    return EvalQuestion(qid, "factual", f"{qid}?", "a", articles, {}, True, **fields)


class Scripted:
    def __init__(self, script: dict[str, list[str]]) -> None:
        self.script = script
        self.depths: list[int] = []

    def retrieve(self, question: str, k: int) -> list[SearchHit]:
        self.depths.append(k)
        return [
            SearchHit(a, f"title {a}", 0, f"chunk {a}", 0.0, f"abstract {a}")
            for a in self.script[question]
        ][:k]


A, B, C, D = "2401.00001", "2402.00002", "2403.00003", "2404.00004"


def test_the_pool_holds_unexpected_results_with_the_rank_each_configuration_gave() -> None:
    vector = Scripted({"q1?": [A, B, C]})
    hybrid = Scripted({"q1?": [C, A, D]})
    items = build_pool([question()], {"vector": vector, "hybrid": hybrid})
    assert [(i.arxiv_id, dict(i.found_by)) for i in items] == [
        (B, {"vector": 2}),
        (C, {"vector": 3, "hybrid": 1}),
        (D, {"hybrid": 3}),
    ]
    assert items[0].title == f"title {B}"
    assert items[0].abstract == f"abstract {B}"


def test_expected_equivalent_and_already_judged_articles_are_not_pooled() -> None:
    q = EvalQuestion(
        "q1",
        "factual",
        "q1?",
        "a",
        (A,),
        {A: (B,)},
        True,
        also_relevant=(C,),
        judged_not_relevant=(D,),
    )
    items = build_pool([q], {"r": Scripted({"q1?": [A, B, C, D, "2405.00005"]})})
    assert [i.arxiv_id for i in items] == ["2405.00005"]


def test_unanswerable_questions_are_not_pooled_and_not_searched() -> None:
    nothing = EvalQuestion("u1", "factual", "u1?", "a", (), {}, True, answerable=False)
    retriever = Scripted({"q1?": [B]})
    items = build_pool([nothing, question()], {"r": retriever})
    assert [i.question_id for i in items] == ["q1"]
    assert retriever.depths == [10]


def test_the_depth_is_passed_to_the_retrievers() -> None:
    retriever = Scripted({"q1?": [B]})
    build_pool([question()], {"r": retriever}, depth=4)
    assert retriever.depths == [4]


def test_review_record_asks_for_a_verdict() -> None:
    (item,) = build_pool([question()], {"r": Scripted({"q1?": [B]})})
    record = review_record(item)
    assert record["relevant"] is None
    assert record["question_id"] == "q1"
    assert record["found_by"] == {"r": 1}
    assert json.loads(json.dumps(record)) == record


def raw(**fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "q1",
        "type": "factual",
        "question": "What?",
        "expected_answer": "That.",
        "expected_articles": [A],
        "validated": True,
    }
    return {**base, **fields}


def verdict(value: object, article: str = B, qid: str = "q1") -> dict[str, Any]:
    return {"question_id": qid, "arxiv_id": article, "relevant": value}


def test_verdicts_are_recorded_in_the_right_lists() -> None:
    records, relevant, not_relevant, pending = apply_judgements(
        [raw()], [verdict(True, B), verdict(False, C), verdict(None, D)]
    )
    assert (relevant, not_relevant, pending) == (1, 1, 1)
    assert records[0]["also_relevant"] == [B]
    assert records[0]["judged_not_relevant"] == [C]
    assert D not in json.dumps(records[0])  # not judged: not stored


def test_applying_twice_adds_nothing_and_keeps_other_fields() -> None:
    first, *_ = apply_judgements([raw(extra="kept")], [verdict(True)])
    second, *_ = apply_judgements(first, [verdict(True)])
    assert second[0]["also_relevant"] == [B]
    assert second[0]["extra"] == "kept"


def test_versioned_ids_are_normalised() -> None:
    records, *_ = apply_judgements([raw()], [verdict(True, "2402.00002v2")])
    assert records[0]["also_relevant"] == [B]


@pytest.mark.parametrize(
    ("judgements", "message"),
    [
        ([verdict(True, qid="nope")], "unknown question 'nope'"),
        ([verdict("yes")], "relevant must be true, false or null"),
        ([verdict(1)], "relevant must be true, false or null"),
        ([verdict(True, A)], "is an expected article"),
        ([verdict(True, B), verdict(False, B)], "judged the other way"),
        ([verdict(False, B), verdict(True, B)], "judged the other way"),
        ([verdict(True, "garbage")], "arXiv identifier"),
    ],
)
def test_inconsistent_verdicts_are_refused(judgements: list[dict[str, Any]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        apply_judgements([raw()], judgements)


def write(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_apply_updates_the_question_file_and_the_result_still_parses(tmp_path: Path) -> None:
    questions, review = tmp_path / "questions.jsonl", tmp_path / "review.jsonl"
    write(questions, [raw()])
    write(review, [verdict(True, B), verdict(False, C), verdict(None, D)])
    assert apply(review, questions) == 0
    (record,) = read_lines(questions)
    parsed = parse_question(record)
    assert parsed.also_relevant == (B,)
    assert parsed.judged_not_relevant == (C,)


def test_apply_writes_nothing_when_a_verdict_is_inconsistent(tmp_path: Path) -> None:
    questions, review = tmp_path / "questions.jsonl", tmp_path / "review.jsonl"
    write(questions, [raw()])
    before = questions.read_text(encoding="utf-8")
    write(review, [verdict(True, B), verdict(True, A)])  # the second is an expected article
    assert apply(review, questions) == 2
    assert questions.read_text(encoding="utf-8") == before


def test_apply_reports_bad_json_with_its_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    questions, review = tmp_path / "questions.jsonl", tmp_path / "review.jsonl"
    write(questions, [raw()])
    review.write_text('{"question_id": "q1"\n', encoding="utf-8")
    assert apply(review, questions) == 2
    assert "review.jsonl:1: invalid JSON" in capsys.readouterr().out


def test_apply_refuses_a_verdict_on_an_equivalent_of_an_expected_article(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    questions, review = tmp_path / "questions.jsonl", tmp_path / "review.jsonl"
    write(questions, [raw(equivalents={A: [B]})])
    before = questions.read_text(encoding="utf-8")
    write(review, [verdict(True, B)])  # B is the same work as A: not a separate judgement
    assert apply(review, questions) == 2
    assert "cannot also be judged" in capsys.readouterr().out
    assert questions.read_text(encoding="utf-8") == before
