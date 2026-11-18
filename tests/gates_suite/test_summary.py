from devgates.report import Finding, GateResult, Report
from devgates.summary import summarize


def failed(gate: str, *findings: Finding) -> GateResult:
    return GateResult(gate=gate, status="failed", findings=findings)


def test_summary_lists_each_failing_gate_with_its_first_findings() -> None:
    # Arrange
    report = Report(
        gates=(
            failed(
                "types", Finding(cause="Incompatible", file="src/a.py", line=12, rule="assignment")
            ),
            failed("lint", Finding(cause="Unused", file="src/b.py", line=3, rule="F401")),
        )
    )

    # Act
    text = summarize(report)

    # Assert
    assert text == (
        "types : failed\n"
        "  src/a.py:12 assignment Incompatible\n"
        "lint : failed\n"
        "  src/b.py:3 F401 Unused"
    )


def test_summary_is_empty_when_every_gate_passes() -> None:
    # Arrange
    report = Report(gates=(GateResult(gate="lint", status="passed"),))

    # Act
    text = summarize(report)

    # Assert
    assert text == ""


def test_summary_omits_passing_gates() -> None:
    # Arrange
    report = Report(
        gates=(GateResult(gate="lint", status="passed"), failed("types", Finding(cause="x")))
    )

    # Act
    text = summarize(report)

    # Assert
    assert text == "types : failed\n  x"


def test_summary_shows_three_findings_then_counts_the_others() -> None:
    # Arrange
    report = Report(gates=(failed("lint", *(Finding(cause=f"m{n}") for n in range(5))),))

    # Act
    text = summarize(report)

    # Assert
    assert text == "lint : failed\n  m0\n  m1\n  m2\n  … et 2 autres"


def test_summary_does_not_count_others_when_exactly_three_findings() -> None:
    # Arrange
    report = Report(gates=(failed("lint", *(Finding(cause=f"m{n}") for n in range(3))),))

    # Act
    text = summarize(report)

    # Assert
    assert text == "lint : failed\n  m0\n  m1\n  m2"


def test_summary_shows_the_error_message_of_a_gate_in_error() -> None:
    # Arrange
    report = Report(gates=(GateResult(gate="mutation", status="error", error="docker absent"),))

    # Act
    text = summarize(report)

    # Assert
    assert text == "mutation : error\n  docker absent"


def test_summary_shows_the_file_without_line_when_the_line_is_unknown() -> None:
    # Arrange
    finding = Finding(cause="lignes non couvertes : 3", file="src/a.py")
    report = Report(gates=(failed("tests", finding),))

    # Act
    text = summarize(report)

    # Assert
    assert text == "tests : failed\n  src/a.py lignes non couvertes : 3"
