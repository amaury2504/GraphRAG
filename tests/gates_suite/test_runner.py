import pytest

from devgates.report import Finding, GateResult
from devgates.runner import Gate, InvalidGateSelectionError, UnknownGateError, run_gates


def recording_gate(name: str, calls: list[str], *, ok: bool = True) -> Gate:
    def run() -> GateResult:
        calls.append(name)
        if ok:
            return GateResult(gate=name, status="passed")
        return GateResult(gate=name, status="failed", findings=(Finding(cause="x"),))

    return Gate(name=name, run=run)


def test_run_gates_runs_every_gate_when_an_early_one_fails() -> None:
    # Arrange
    calls: list[str] = []
    gates = [recording_gate("lint", calls, ok=False), recording_gate("types", calls)]

    # Act
    run_gates(gates)

    # Assert
    assert calls == ["lint", "types"]


def test_run_gates_runs_gates_in_declared_order() -> None:
    # Arrange
    calls: list[str] = []
    gates = [recording_gate(name, calls) for name in ("lint", "format", "types")]

    # Act
    report = run_gates(gates)

    # Assert
    assert [r.gate for r in report.gates] == ["lint", "format", "types"]


def test_run_gates_runs_only_selected_gate_when_only_is_given() -> None:
    # Arrange
    calls: list[str] = []
    gates = [recording_gate("lint", calls), recording_gate("types", calls)]

    # Act
    run_gates(gates, only="types")

    # Assert
    assert calls == ["types"]


def test_run_gates_raises_when_gate_name_is_unknown() -> None:
    # Arrange
    gates = [recording_gate("lint", [])]

    # Act / Assert
    with pytest.raises(UnknownGateError, match="nope"):
        run_gates(gates, only="nope")


def test_run_gates_skips_gate_when_skip_is_given() -> None:
    # Arrange
    calls: list[str] = []
    gates = [recording_gate("lint", calls), recording_gate("mutation", calls)]

    # Act
    run_gates(gates, skip=["mutation"])

    # Assert
    assert calls == ["lint"]


def test_run_gates_skips_every_named_gate_when_several_skips_are_given() -> None:
    # Arrange
    calls: list[str] = []
    gates = [recording_gate(name, calls) for name in ("lint", "tests", "mutation")]

    # Act
    run_gates(gates, skip=["mutation", "tests"])

    # Assert
    assert calls == ["lint"]


def test_run_gates_keeps_declared_order_when_some_gates_are_skipped() -> None:
    # Arrange
    calls: list[str] = []
    gates = [recording_gate(name, calls) for name in ("lint", "format", "types", "tests")]

    # Act
    run_gates(gates, skip=["format"])

    # Assert
    assert calls == ["lint", "types", "tests"]


def test_run_gates_report_lists_only_the_gates_that_ran_when_some_are_skipped() -> None:
    # Arrange
    gates = [recording_gate(name, []) for name in ("lint", "mutation")]

    # Act
    report = run_gates(gates, skip=["mutation"])

    # Assert
    assert [r.gate for r in report.gates] == ["lint"]


def test_run_gates_raises_when_a_skipped_gate_name_is_unknown() -> None:
    # Arrange
    gates = [recording_gate("lint", [])]

    # Act / Assert
    with pytest.raises(UnknownGateError, match="mutaton"):
        run_gates(gates, skip=["mutaton"])


def test_run_gates_raises_when_every_gate_is_skipped() -> None:
    # Arrange
    gates = [recording_gate(name, []) for name in ("lint", "types")]

    # Act / Assert
    with pytest.raises(InvalidGateSelectionError) as exc:
        run_gates(gates, skip=["lint", "types"])

    # Assert
    assert str(exc.value) == "aucun gate à exécuter : tous sont ignorés"


def test_run_gates_raises_when_only_and_skip_are_both_given() -> None:
    # Arrange
    gates = [recording_gate(name, []) for name in ("lint", "types")]

    # Act / Assert
    with pytest.raises(InvalidGateSelectionError) as exc:
        run_gates(gates, only="lint", skip=["types"])

    # Assert
    assert str(exc.value) == "--only et --skip ensemble : choisir l'un ou l'autre"


@pytest.mark.parametrize(
    "exception",
    [
        pytest.param(FileNotFoundError("ruff"), id="command-not-found"),
        pytest.param(RuntimeError("boom"), id="unexpected-exception"),
    ],
)
def test_run_gates_marks_gate_in_error_when_it_raises(exception: Exception) -> None:
    # Arrange
    calls: list[str] = []

    def broken() -> GateResult:
        raise exception

    gates = [Gate(name="lint", run=broken), recording_gate("types", calls)]

    # Act
    report = run_gates(gates)

    # Assert
    assert [(r.gate, r.status, r.error) for r in report.gates] == [
        ("lint", "error", str(exception)),
        ("types", "passed", None),
    ]
