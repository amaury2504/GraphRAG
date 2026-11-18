from devgates.mutation import CommandOutput, run_mutation
from devgates.report import Finding
from tests.gates_suite.test_mutation.fakes import (
    NO_SURVIVOR,
    RUN_OK,
    SURVIVOR_DIFF,
    SURVIVOR_LISTING,
    SURVIVOR_NAME,
    FakeExecutor,
)

RUN = ("mutmut", "run")
RESULTS = ("mutmut", "results")
SHOW = ("mutmut", "show", SURVIVOR_NAME)


def with_one_survivor() -> FakeExecutor:
    return FakeExecutor(
        {RUN: RUN_OK, RESULTS: SURVIVOR_LISTING, SHOW: CommandOutput(0, SURVIVOR_DIFF)}
    )


def test_run_mutation_passes_when_no_mutant_survives() -> None:
    # Arrange
    execute = FakeExecutor({RUN: RUN_OK, RESULTS: NO_SURVIVOR})

    # Act
    result = run_mutation(execute)

    # Assert
    assert (result.gate, result.status, result.findings) == ("mutation", "passed", ())


def test_run_mutation_fails_when_a_mutant_survives() -> None:
    # Arrange
    execute = with_one_survivor()

    # Act
    result = run_mutation(execute)

    # Assert
    assert result.status == "failed"


def test_run_mutation_reports_error_when_mutmut_run_fails() -> None:
    # Arrange
    execute = FakeExecutor({RUN: CommandOutput(returncode=1, stdout="boom")})

    # Act
    result = run_mutation(execute)

    # Assert
    assert (result.gate, result.status, result.error) == (
        "mutation",
        "error",
        "mutmut run a échoué (code 1) : boom",
    )


def test_run_mutation_does_not_read_results_when_mutmut_run_fails() -> None:
    # Arrange
    execute = FakeExecutor({RUN: CommandOutput(returncode=1, stdout="boom")})

    # Act
    run_mutation(execute)

    # Assert
    assert execute.calls == [RUN]


def test_run_mutation_reports_error_when_mutmut_results_fails() -> None:
    # Arrange
    execute = FakeExecutor({RUN: RUN_OK, RESULTS: CommandOutput(returncode=2, stdout="")})

    # Act
    result = run_mutation(execute)

    # Assert
    assert (result.gate, result.status, result.error) == (
        "mutation",
        "error",
        "mutmut results a échoué (code 2) : ",
    )


def test_run_mutation_reports_one_finding_per_surviving_mutant() -> None:
    # Arrange
    second = "app.x_other__mutmut_2"
    listing = CommandOutput(0, f"    {SURVIVOR_NAME}: survived\n    {second}: survived\n")
    execute = FakeExecutor(
        {
            RUN: RUN_OK,
            RESULTS: listing,
            SHOW: CommandOutput(0, SURVIVOR_DIFF),
            ("mutmut", "show", second): CommandOutput(0, SURVIVOR_DIFF),
        }
    )

    # Act
    result = run_mutation(execute)

    # Assert
    assert [f.mutant for f in result.findings] == [SURVIVOR_NAME, second]


def test_run_mutation_finding_has_no_file_when_diff_has_no_file_header() -> None:
    # Arrange
    execute = FakeExecutor(
        {
            RUN: RUN_OK,
            RESULTS: SURVIVOR_LISTING,
            SHOW: CommandOutput(0, "diff illisible"),
        }
    )

    # Act
    result = run_mutation(execute)

    # Assert
    assert result.findings[0].file is None


def test_run_mutation_ignores_listing_lines_without_status_when_output_has_noise() -> None:
    # Arrange
    noisy = CommandOutput(0, f"Avertissement parasite\n\n    {SURVIVOR_NAME}: survived\n")
    execute = FakeExecutor({RUN: RUN_OK, RESULTS: noisy, SHOW: CommandOutput(0, SURVIVOR_DIFF)})

    # Act
    result = run_mutation(execute)

    # Assert
    assert [f.mutant for f in result.findings] == [SURVIVOR_NAME]


def test_run_mutation_keeps_whole_status_when_status_contains_a_colon() -> None:
    # Arrange
    listing = CommandOutput(0, f"    {SURVIVOR_NAME}: survived: extra\n")
    execute = FakeExecutor({RUN: RUN_OK, RESULTS: listing, SHOW: CommandOutput(0, SURVIVOR_DIFF)})

    # Act
    result = run_mutation(execute)

    # Assert
    assert result.findings[0].cause == "mutant survived: extra"


def test_run_mutation_finding_contains_mutant_name_source_file_and_diff() -> None:
    # Arrange
    execute = with_one_survivor()

    # Act
    result = run_mutation(execute)

    # Assert
    assert result.findings == (
        Finding(
            cause="mutant survived",
            mutant=SURVIVOR_NAME,
            file="src/app/__init__.py",
            diff=SURVIVOR_DIFF,
        ),
    )


def test_run_mutation_passes_patterns_to_mutmut_run_when_targets_are_given() -> None:
    # Arrange
    targeted_run = ("mutmut", "run", "app.journal.values.*", "devgates.report.*")
    execute = FakeExecutor({targeted_run: RUN_OK, RESULTS: NO_SURVIVOR})

    # Act
    result = run_mutation(execute, targets=("app.journal.values.*", "devgates.report.*"))

    # Assert
    assert (result.status, execute.calls[0]) == ("passed", targeted_run)


def test_run_mutation_runs_everything_when_no_targets_are_given() -> None:
    # Arrange
    execute = FakeExecutor({RUN: RUN_OK, RESULTS: NO_SURVIVOR})

    # Act
    run_mutation(execute, targets=None)

    # Assert
    assert execute.calls[0] == RUN


def test_run_mutation_passes_without_running_mutmut_when_targets_are_empty() -> None:
    # Arrange
    execute = FakeExecutor({})

    # Act
    result = run_mutation(execute, targets=())

    # Assert
    assert (result.gate, result.status, execute.calls) == ("mutation", "passed", [])


def test_run_mutation_ignores_unchecked_mutants_outside_targets_when_targets_are_given() -> None:
    # Arrange
    listing = CommandOutput(
        0,
        "    app.other.x_f__mutmut_1: not checked\n    app.mine.x_g__mutmut_1: survived\n",
    )
    show = ("mutmut", "show", "app.mine.x_g__mutmut_1")
    execute = FakeExecutor(
        {
            ("mutmut", "run", "app.mine.*"): RUN_OK,
            RESULTS: listing,
            show: CommandOutput(0, SURVIVOR_DIFF),
        }
    )

    # Act
    result = run_mutation(execute, targets=("app.mine.*",))

    # Assert
    assert [finding.mutant for finding in result.findings] == ["app.mine.x_g__mutmut_1"]
