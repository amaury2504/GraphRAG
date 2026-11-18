import json

import pytest

from devgates.report import Finding, GateResult, Report


def passed(name: str) -> GateResult:
    return GateResult(gate=name, status="passed")


def failed(name: str, *findings: Finding) -> GateResult:
    return GateResult(gate=name, status="failed", findings=findings)


def in_error(name: str) -> GateResult:
    return GateResult(gate=name, status="error", error="outil indisponible")


def test_report_status_is_passed_when_all_gates_passed() -> None:
    # Arrange
    report = Report(gates=(passed("lint"), passed("types")))

    # Act
    status = report.status

    # Assert
    assert status == "passed"


def test_report_status_is_failed_when_a_gate_failed() -> None:
    # Arrange
    report = Report(gates=(passed("lint"), failed("types", Finding(cause="x"))))

    # Act
    status = report.status

    # Assert
    assert status == "failed"


def test_report_status_is_error_when_a_gate_is_in_error() -> None:
    # Arrange
    report = Report(gates=(passed("lint"), in_error("mutation")))

    # Act
    status = report.status

    # Assert
    assert status == "error"


def test_report_status_is_error_when_a_gate_failed_and_another_is_in_error() -> None:
    # Arrange
    report = Report(gates=(failed("lint", Finding(cause="x")), in_error("mutation")))

    # Act
    status = report.status

    # Assert
    assert status == "error"


def test_report_exit_code_is_zero_when_passed() -> None:
    # Arrange
    report = Report(gates=(passed("lint"),))

    # Act
    exit_code = report.exit_code

    # Assert
    assert exit_code == 0


@pytest.mark.parametrize(
    "gates",
    [
        pytest.param((failed("lint", Finding(cause="x")),), id="failed"),
        pytest.param((in_error("lint"),), id="error"),
    ],
)
def test_report_exit_code_is_one_when_failed_or_error(gates: tuple[GateResult, ...]) -> None:
    # Arrange
    report = Report(gates=gates)

    # Act
    exit_code = report.exit_code

    # Assert
    assert exit_code == 1


def test_report_keeps_one_entry_per_gate_when_several_gates_fail() -> None:
    # Arrange
    report = Report(gates=(failed("lint", Finding(cause="a")), failed("types", Finding(cause="b"))))

    # Act
    names = [gate["gate"] for gate in report.to_dict()["gates"]]

    # Assert
    assert names == ["lint", "types"]


def test_report_keeps_one_finding_per_gate_when_same_location_fails_two_gates() -> None:
    # Arrange
    report = Report(
        gates=(
            failed("lint", Finding(cause="F401", file="src/app/x.py", line=3)),
            failed("types", Finding(cause="arg-type", file="src/app/x.py", line=3)),
        )
    )

    # Act
    findings_per_gate = [len(gate["findings"]) for gate in report.to_dict()["gates"]]

    # Assert
    assert findings_per_gate == [1, 1]


def test_report_lists_every_finding_when_a_gate_has_several_locations() -> None:
    # Arrange
    report = Report(
        gates=(
            failed(
                "lint",
                Finding(cause="F401", file="src/app/a.py", line=1),
                Finding(cause="F401", file="src/app/b.py", line=7),
            ),
        )
    )

    # Act
    locations = [(f["file"], f["line"]) for f in report.to_dict()["gates"][0]["findings"]]

    # Assert
    assert locations == [("src/app/a.py", 1), ("src/app/b.py", 7)]


def test_report_allows_missing_file_and_line_when_gate_has_no_location() -> None:
    # Arrange
    report = Report(gates=(failed("tests", Finding(cause="code de sortie 1")),))

    # Act
    finding = report.to_dict()["gates"][0]["findings"][0]

    # Assert
    assert (finding["file"], finding["line"]) == (None, None)


def test_report_serializes_error_rule_mutant_and_diff_when_present() -> None:
    # Arrange
    finding = Finding(cause="survivant", rule="E501", mutant="m_1", diff="-a\n+b")
    report = Report(
        gates=(
            in_error("mutation"),
            failed("lint", finding),
        )
    )

    # Act
    gates = report.to_dict()["gates"]

    # Assert
    got = gates[1]["findings"][0]
    assert (gates[0]["error"], got["rule"], got["mutant"], got["diff"]) == (
        "outil indisponible",
        "E501",
        "m_1",
        "-a\n+b",
    )


def test_report_to_json_keeps_accents_unescaped_when_dumped() -> None:
    # Arrange
    report = Report(gates=(failed("lint", Finding(cause="échec")),))

    # Act
    document = report.to_json()

    # Assert
    assert "échec" in document


def test_report_to_json_is_indented_by_two_spaces_when_dumped() -> None:
    # Arrange
    report = Report(gates=(passed("lint"),))

    # Act
    document = report.to_json()

    # Assert
    assert document.startswith('{\n  "schema_version": 1,')


def test_report_serializes_to_json_with_schema_version_when_dumped() -> None:
    # Arrange
    report = Report(gates=(failed("lint", Finding(cause="F401", file="a.py", line=1)),))

    # Act
    document = json.loads(report.to_json())

    # Assert
    assert document == {
        "schema_version": 1,
        "status": "failed",
        "exit_code": 1,
        "gates": [
            {
                "gate": "lint",
                "status": "failed",
                "error": None,
                "findings": [
                    {
                        "cause": "F401",
                        "file": "a.py",
                        "line": 1,
                        "rule": None,
                        "mutant": None,
                        "diff": None,
                    }
                ],
            }
        ],
    }
