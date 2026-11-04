"""Evaluation questions stored as JSON Lines: one JSON object per line."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from graphrag.domain.evaluation import EvalQuestion, check_unique_ids, parse_question

DEFAULT_PATH = Path("eval/questions.jsonl")
CANDIDATES_PATH = Path("eval/candidates.jsonl")


def read_questions(path: Path = DEFAULT_PATH) -> list[EvalQuestion]:
    questions = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError("a line must be a JSON object")
            questions.append(parse_question(data))
        except ValueError as exc:  # includes json.JSONDecodeError
            raise ValueError(f"{path}:{number}: {exc}") from None
    check_unique_ids(questions)
    return questions


def write_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write one record per line, readable by `read_questions` (extra keys are allowed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
