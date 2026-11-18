import pytest

from devgates.mutation import CommandOutput
from devgates.targets import mutation_targets, select_mutation_targets
from tests.gates_suite.test_mutation.fakes import FakeExecutor


def test_select_mutation_targets_returns_patterns_for_changed_source_files() -> None:
    # Arrange
    changed = "src/app/journal/values.py\nsrc/app/gates/report.py\n"

    # Act
    targets = select_mutation_targets(changed, untracked="")

    # Assert
    assert targets == ("app.journal.values.*", "app.gates.report.*")


def test_select_mutation_targets_ignores_files_outside_src_or_not_python() -> None:
    # Arrange
    changed = "docs/regles.md\ntests/journal/test_event.py\nsrc/app/py.typed\nscripts/gates.py\n"

    # Act
    targets = select_mutation_targets(changed, untracked="")

    # Assert
    assert targets == ()


def test_select_mutation_targets_includes_untracked_new_files() -> None:
    # Arrange
    untracked = "src/app/journal/new_module.py\n"

    # Act
    targets = select_mutation_targets("", untracked=untracked)

    # Assert
    assert targets == ("app.journal.new_module.*",)


def test_select_mutation_targets_returns_empty_when_no_source_changed() -> None:
    # Arrange / Act
    targets = select_mutation_targets("", untracked="")

    # Assert
    assert targets == ()


def test_select_mutation_targets_deduplicates_files_listed_twice() -> None:
    # Arrange
    changed = "src/app/journal/values.py\n"
    untracked = "src/app/journal/values.py\n"

    # Act
    targets = select_mutation_targets(changed, untracked)

    # Assert
    assert targets == ("app.journal.values.*",)


DIFF = ("git", "diff", "--name-only", "main")
UNTRACKED = ("git", "ls-files", "--others", "--exclude-standard")
GIT_OK = CommandOutput(0, "")
GIT_FAILED = CommandOutput(128, "fatal")


@pytest.mark.parametrize(
    ("diff", "untracked"),
    [
        pytest.param(GIT_FAILED, GIT_OK, id="diff-echoue"),
        pytest.param(GIT_OK, GIT_FAILED, id="liste-echoue"),
        pytest.param(GIT_FAILED, GIT_FAILED, id="les-deux-echouent"),
    ],
)
def test_mutation_targets_fall_back_to_full_run_when_git_fails(
    diff: CommandOutput, untracked: CommandOutput
) -> None:
    # Arrange
    git = FakeExecutor({DIFF: diff, UNTRACKED: untracked})

    # Act
    targets = mutation_targets(git, base="main")

    # Assert
    assert targets is None


def test_mutation_targets_combine_diff_and_untracked_files_when_git_succeeds() -> None:
    # Arrange
    git = FakeExecutor(
        {DIFF: CommandOutput(0, "src/app/a.py\n"), UNTRACKED: CommandOutput(0, "src/app/b.py\n")}
    )

    # Act
    targets = mutation_targets(git, base="main")

    # Assert
    assert targets == ("app.a.*", "app.b.*")
