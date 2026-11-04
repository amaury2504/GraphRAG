import json
from pathlib import Path

import pytest

from graphrag.adapters.question_file import read_questions, write_records

GOOD = {
    "id": "q1",
    "type": "factual",
    "question": "Qu'est-ce que le RAG ? ✓",
    "expected_answer": "Une méthode.",
    "expected_articles": ["2401.00001"],
    "validated": True,
}


def test_roundtrip_keeps_unicode_and_extra_keys(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    write_records(path, [{**GOOD, "source_titles": ["Titre"]}, {**GOOD, "id": "q2"}])
    text = path.read_text(encoding="utf-8")
    assert "Qu'est-ce que le RAG ? ✓" in text  # not escaped to \u sequences
    assert text.endswith("\n")
    assert len(text.splitlines()) == 2
    questions = read_questions(path)
    assert [q.id for q in questions] == ["q1", "q2"]
    assert questions[0].question == "Qu'est-ce que le RAG ? ✓"
    assert questions[0].validated is True


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(f"\n{json.dumps(GOOD)}\n\n  \n", encoding="utf-8")
    assert len(read_questions(path)) == 1


def test_empty_file_means_no_questions(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    write_records(path, [])
    assert path.read_text(encoding="utf-8") == ""
    assert read_questions(path) == []


def test_writing_creates_missing_folders(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "er" / "q.jsonl"
    write_records(path, [GOOD])
    assert path.exists()


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("{not json", r"q\.jsonl:2: "),
        ("[1, 2]", r"q\.jsonl:2: a line must be a JSON object"),
        (json.dumps({**GOOD, "id": "q9", "type": "weird"}), r"q\.jsonl:2: q9: type must be one of"),
    ],
)
def test_errors_name_the_file_and_line(tmp_path: Path, line: str, message: str) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(f"{json.dumps(GOOD)}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_questions(path)


def test_duplicate_ids_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    write_records(path, [GOOD, GOOD])
    with pytest.raises(ValueError, match="duplicate question id q1"):
        read_questions(path)
