import ast
from pathlib import Path

import pytest

from devgates.bypass import run_bypass


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_bypass_gate_passes_when_no_forbidden_call_exists(tmp_path: Path) -> None:
    # Arrange
    clean = write(tmp_path, "clean.py", "value = 1\nother = value  # type: ignore[misc]\n")

    # Act
    result = run_bypass([clean])

    # Assert
    assert (result.gate, result.status, result.findings) == ("bypass", "passed", ())


def test_bypass_gate_reports_file_and_line_when_model_construct_is_called(tmp_path: Path) -> None:
    # Arrange
    bad = write(tmp_path, "bad.py", "a = 1\nM.model_construct(x=1)\n")

    # Act
    result = run_bypass([bad])

    # Assert
    assert [(f.file, f.line, f.rule, f.cause) for f in result.findings] == [
        (str(bad), 2, "BYPASS-MODEL-CONSTRUCT", "model_construct contourne la validation")
    ]


def test_bypass_gate_reports_when_model_copy_receives_update(tmp_path: Path) -> None:
    # Arrange
    bad = write(tmp_path, "bad.py", 'event.model_copy(update={"a": 1})\n')

    # Act
    result = run_bypass([bad])

    # Assert
    assert [(result.gate, result.status, f.line, f.rule) for f in result.findings] == [
        ("bypass", "failed", 1, "BYPASS-MODEL-COPY-UPDATE")
    ]


def test_bypass_gate_accepts_model_copy_without_update(tmp_path: Path) -> None:
    # Arrange
    good = write(tmp_path, "good.py", "event.model_copy(deep=True)\n")

    # Act
    result = run_bypass([good])

    # Assert
    assert result.status == "passed"


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("from typing import cast\nx = cast(int, y)\n", id="nom"),
        pytest.param("import typing\nx = typing.cast(int, y)\n", id="attribut"),
    ],
)
def test_bypass_gate_reports_when_typing_cast_is_used(tmp_path: Path, source: str) -> None:
    # Arrange
    bad = write(tmp_path, "bad.py", source)

    # Act
    result = run_bypass([bad])

    # Assert
    assert [(f.line, f.rule) for f in result.findings] == [(2, "BYPASS-CAST")]


@pytest.mark.parametrize(
    "comment",
    [
        pytest.param("# type: ignore", id="espace"),
        pytest.param("# type:ignore", id="colle"),
    ],
)
def test_bypass_gate_reports_when_type_ignore_has_no_error_code(
    tmp_path: Path, comment: str
) -> None:
    # Arrange
    bad = write(tmp_path, "bad.py", f"x = 1  {comment}\n")

    # Act
    result = run_bypass([bad])

    # Assert
    assert [(f.line, f.rule) for f in result.findings] == [(1, "BYPASS-TYPE-IGNORE")]


def test_bypass_gate_reports_every_occurrence_when_several_files_are_wrong(
    tmp_path: Path,
) -> None:
    # Arrange
    first = write(tmp_path, "a.py", "M.model_construct()\ny = cast(int, z)\n")
    second = write(tmp_path, "b.py", "x = 1  # type: ignore\n")

    # Act
    result = run_bypass([first, second])

    # Assert
    assert [(f.file, f.line, f.rule) for f in result.findings] == [
        (str(first), 1, "BYPASS-MODEL-CONSTRUCT"),
        (str(first), 2, "BYPASS-CAST"),
        (str(second), 1, "BYPASS-TYPE-IGNORE"),
    ]


def test_bypass_gate_reports_error_when_a_source_file_cannot_be_parsed(tmp_path: Path) -> None:
    # Arrange
    broken = write(tmp_path, "broken.py", "def (:\n")
    with pytest.raises(SyntaxError) as expected:
        ast.parse("def (:\n", filename=str(broken))

    # Act
    result = run_bypass([broken])

    # Assert
    assert (result.gate, result.status, result.error) == (
        "bypass",
        "error",
        f"{broken} : {expected.value}",
    )


def test_bypass_gate_ignores_forbidden_names_in_strings_and_comments(tmp_path: Path) -> None:
    # Arrange
    quiet = write(
        tmp_path,
        "quiet.py",
        'text = "x.model_construct() # type: ignore"\n# model_construct, cast(int, y)\n',
    )

    # Act
    result = run_bypass([quiet])

    # Assert
    assert result.status == "passed"
