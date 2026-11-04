from pathlib import Path

import pytest

from graphrag.cli.generate_questions import MAX_WITHOUT_CONFIRMATION, check_request


def test_the_first_batch_size_is_eight_questions() -> None:
    assert MAX_WITHOUT_CONFIRMATION == 8


def test_two_per_type_is_allowed_without_confirmation(tmp_path: Path) -> None:
    assert check_request(2, False, tmp_path / "c.jsonl", False) is None


def test_a_bigger_batch_needs_the_cost_confirmation(tmp_path: Path) -> None:
    message = check_request(3, False, tmp_path / "c.jsonl", False)
    assert message is not None
    assert "12 questions means 12 LLM calls" in message
    assert "--confirm-cost" in message
    assert check_request(3, True, tmp_path / "c.jsonl", False) is None


@pytest.mark.parametrize("per_type", [0, -1])
def test_per_type_must_be_positive(per_type: int, tmp_path: Path) -> None:
    assert check_request(per_type, True, tmp_path / "c.jsonl", False) == (
        "--per-type must be at least 1"
    )


def test_an_existing_file_is_not_overwritten_unless_forced(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text("x", encoding="utf-8")
    message = check_request(2, False, path, False)
    assert message is not None
    assert "already exists" in message
    assert check_request(2, False, path, True) is None
