from devgates.mutation import CommandOutput
from devgates.parsers import ParseError
from devgates.report import Finding, GateResult
from devgates.tool_gate import run_tool_gate

FINDING = Finding(cause="boom", file="a.py", line=1, rule="X")


def parse_one(_text: str) -> tuple[Finding, ...]:
    return (FINDING,)


def parse_none(_text: str) -> tuple[Finding, ...]:
    return ()


def parse_fails(_text: str) -> tuple[Finding, ...]:
    raise ParseError("sortie de x illisible")


def test_tool_gate_passes_when_the_exit_code_is_zero() -> None:
    # Arrange
    output = CommandOutput(returncode=0, stdout="")

    # Act
    result = run_tool_gate("lint", output, parse_fails)

    # Assert
    assert result == GateResult(gate="lint", status="passed")


def test_tool_gate_fails_with_the_parsed_findings_when_the_exit_code_is_not_zero() -> None:
    # Arrange
    output = CommandOutput(returncode=1, stdout="texte")

    # Act
    result = run_tool_gate("lint", output, parse_one)

    # Assert
    assert result == GateResult(gate="lint", status="failed", findings=(FINDING,))


def test_tool_gate_gives_the_output_to_the_parser() -> None:
    # Arrange
    received: list[str] = []

    def parse(text: str) -> tuple[Finding, ...]:
        received.append(text)
        return (FINDING,)

    # Act
    run_tool_gate("lint", CommandOutput(returncode=1, stdout="le texte"), parse)

    # Assert
    assert received == ["le texte"]


def test_tool_gate_falls_back_to_the_exit_code_and_output_tail_when_no_finding_is_parsed() -> None:
    # Arrange
    stdout = "\n".join(f"ligne {n}" for n in range(1, 16))
    output = CommandOutput(returncode=3, stdout=stdout)

    # Act
    result = run_tool_gate("tests", output, parse_none)

    # Assert
    expected_tail = "\n".join(f"ligne {n}" for n in range(6, 16))
    assert result == GateResult(
        gate="tests",
        status="failed",
        findings=(Finding(cause=f"code de sortie 3 : {expected_tail}"),),
    )


def test_tool_gate_reports_error_when_the_output_cannot_be_parsed() -> None:
    # Arrange
    output = CommandOutput(returncode=1, stdout="n'importe quoi")

    # Act
    result = run_tool_gate("types", output, parse_fails)

    # Assert
    assert result == GateResult(gate="types", status="error", error="sortie de x illisible")
