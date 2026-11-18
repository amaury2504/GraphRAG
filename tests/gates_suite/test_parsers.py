import json

import pytest

from devgates.parsers import (
    ParseError,
    parse_coverage,
    parse_junit,
    parse_mypy,
    parse_ruff,
)
from devgates.report import Finding

ROOT = "C:\\Dev\\SMA"


def ruff_violation(filename: str, row: int, code: str, message: str) -> dict[str, object]:
    return {
        "filename": filename,
        "location": {"row": row, "column": 1},
        "code": code,
        "message": message,
    }


def test_parse_ruff_reports_file_line_rule_and_message_for_each_violation() -> None:
    # Arrange
    text = json.dumps(
        [
            ruff_violation("C:\\Dev\\SMA\\src\\app\\a.py", 3, "F401", "`os` imported but unused"),
            ruff_violation("C:/Dev/SMA/src/app/b.py", 9, "E501", "Line too long"),
        ]
    )

    # Act
    findings = parse_ruff(text, ROOT)

    # Assert
    assert findings == (
        Finding(cause="`os` imported but unused", file="src/app/a.py", line=3, rule="F401"),
        Finding(cause="Line too long", file="src/app/b.py", line=9, rule="E501"),
    )


def test_parse_ruff_keeps_a_path_outside_the_root_unchanged() -> None:
    # Arrange
    text = json.dumps([ruff_violation("D:\\ailleurs\\a.py", 1, "F401", "m")])

    # Act
    findings = parse_ruff(text, ROOT)

    # Assert
    assert findings[0].file == "D:/ailleurs/a.py"


def test_parse_ruff_reports_a_file_to_reformat_with_the_rule_unformatted() -> None:
    # Arrange
    text = json.dumps(
        [ruff_violation("src/app/a.py", 2, "unformatted", "File would be reformatted")]
    )

    # Act
    findings = parse_ruff(text, ROOT)

    # Assert
    assert findings == (
        Finding(cause="File would be reformatted", file="src/app/a.py", line=2, rule="unformatted"),
    )


def test_parse_ruff_returns_no_finding_when_the_output_is_an_empty_list() -> None:
    # Arrange / Act
    findings = parse_ruff("[]", ROOT)

    # Assert
    assert findings == ()


@pytest.mark.parametrize("text", ["pas du json", "{}"], ids=["not-json", "not-a-list"])
def test_parse_ruff_raises_when_the_output_is_not_json(text: str) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ParseError, match="sortie de ruff illisible"):
        parse_ruff(text, ROOT)


def test_parse_ruff_names_the_expected_shape_when_the_output_is_not_a_list() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ParseError, match="sortie de ruff illisible : liste attendue"):
        parse_ruff("{}", ROOT)


def mypy_line(severity: str, code: str = "assignment") -> str:
    return json.dumps(
        {
            "file": "src\\app\\a.py",
            "line": 4,
            "message": "Incompatible types",
            "code": code,
            "severity": severity,
        }
    )


def test_parse_mypy_reports_file_line_code_and_message_for_each_error() -> None:
    # Arrange
    text = mypy_line("error") + "\n" + mypy_line("error", "arg-type") + "\n"

    # Act
    findings = parse_mypy(text, ROOT)

    # Assert
    assert findings == (
        Finding(cause="Incompatible types", file="src/app/a.py", line=4, rule="assignment"),
        Finding(cause="Incompatible types", file="src/app/a.py", line=4, rule="arg-type"),
    )


def test_parse_mypy_ignores_notes() -> None:
    # Arrange
    text = mypy_line("note") + "\n" + mypy_line("error") + "\n"

    # Act
    findings = parse_mypy(text, ROOT)

    # Assert
    assert [finding.rule for finding in findings] == ["assignment"]


def test_parse_mypy_raises_when_a_line_is_not_json() -> None:
    # Arrange
    text = mypy_line("error") + "\nTraceback (most recent call last)\n"

    # Act / Assert
    with pytest.raises(ParseError, match="sortie de mypy illisible"):
        parse_mypy(text, ROOT)


def junit(*cases: str) -> str:
    return f'<testsuites><testsuite name="pytest">{"".join(cases)}</testsuite></testsuites>'


def test_parse_junit_reports_each_failed_test_with_its_file_and_message() -> None:
    # Arrange
    text = junit(
        '<testcase classname="tests.a" name="test_ok" file="tests/a.py"/>',
        '<testcase classname="tests.a" name="test_ko" file="tests/a.py">'
        '<failure message="assert 1 == 2">trace</failure></testcase>',
    )

    # Act
    findings = parse_junit(text)

    # Assert
    assert findings == (Finding(cause="test_ko : assert 1 == 2", file="tests/a.py"),)


def test_parse_junit_reports_collection_errors_like_failures() -> None:
    # Arrange
    text = junit(
        '<testcase classname="" name="tests.b"><error message="collection failure">x</error>'
        "</testcase>"
    )

    # Act
    findings = parse_junit(text)

    # Assert
    assert findings == (Finding(cause="tests.b : collection failure"),)


def test_parse_junit_returns_no_finding_when_every_test_passes() -> None:
    # Arrange
    text = junit('<testcase classname="tests.a" name="test_ok" file="tests/a.py"/>')

    # Act
    findings = parse_junit(text)

    # Assert
    assert findings == ()


def test_parse_junit_reports_a_failure_without_message() -> None:
    # Arrange
    text = junit('<testcase classname="t" name="test_ko" file="t.py"><failure/></testcase>')

    # Act
    findings = parse_junit(text)

    # Assert
    assert findings == (Finding(cause="test_ko : ", file="t.py"),)


def test_parse_junit_raises_when_the_output_is_not_xml() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ParseError, match="sortie de pytest illisible"):
        parse_junit("pas du xml")


def coverage(files: dict[str, dict[str, object]]) -> str:
    return json.dumps({"files": files})


def test_parse_coverage_reports_one_finding_per_file_with_its_missing_lines() -> None:
    # Arrange
    text = coverage(
        {
            "src/app/a.py": {"missing_lines": [3, 5], "missing_branches": []},
            "src/app/b.py": {"missing_lines": [8], "missing_branches": []},
        }
    )

    # Act
    findings = parse_coverage(text)

    # Assert
    assert findings == (
        Finding(cause="lignes non couvertes : 3, 5", file="src/app/a.py"),
        Finding(cause="lignes non couvertes : 8", file="src/app/b.py"),
    )


def test_parse_coverage_ignores_files_without_missing_lines() -> None:
    # Arrange
    text = coverage({"src/app/a.py": {"missing_lines": [], "missing_branches": []}})

    # Act
    findings = parse_coverage(text)

    # Assert
    assert findings == ()


@pytest.mark.parametrize(
    ("lines", "branches", "cause"),
    [
        pytest.param([], [[5, 7]], "branches non couvertes : 5→7", id="branches-only"),
        pytest.param(
            [3],
            [[5, 7]],
            "lignes non couvertes : 3 ; branches non couvertes : 5→7",
            id="lines-and-branches",
        ),
        pytest.param(
            [], [[5, -1], [9, 10]], "branches non couvertes : 5→sortie, 9→10", id="exit-branch"
        ),
        pytest.param([], [[5, 0]], "branches non couvertes : 5→0", id="target-zero"),
    ],
)
def test_parse_coverage_reports_missing_branches_of_a_file(
    lines: list[int], branches: list[list[int]], cause: str
) -> None:
    # Arrange
    text = coverage({"src/app/a.py": {"missing_lines": lines, "missing_branches": branches}})

    # Act
    findings = parse_coverage(text)

    # Assert
    assert findings == (Finding(cause=cause, file="src/app/a.py"),)


@pytest.mark.parametrize(
    "text", ["pas du json", "[]", '{"files": []}'], ids=["not-json", "list", "bad-files"]
)
def test_parse_coverage_raises_when_the_output_is_not_json(text: str) -> None:
    # Arrange / Act / Assert
    with pytest.raises(ParseError, match="sortie de coverage illisible"):
        parse_coverage(text)


def test_parse_junit_uses_forward_slashes_in_paths() -> None:
    # Arrange
    case = '<testcase classname="t" name="test_ko" file="tests\\a.py"><failure message="m"/>'
    text = junit(case + "</testcase>")

    # Act
    findings = parse_junit(text)

    # Assert
    assert findings[0].file == "tests/a.py"


def test_parse_coverage_uses_forward_slashes_in_paths() -> None:
    # Arrange
    text = coverage({"src\\app\\a.py": {"missing_lines": [1], "missing_branches": []}})

    # Act
    findings = parse_coverage(text)

    # Assert
    assert findings[0].file == "src/app/a.py"
