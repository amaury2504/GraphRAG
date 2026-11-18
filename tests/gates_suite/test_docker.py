from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from devgates.docker import (
    INSTALL,
    READ_HASH,
    REFRESH,
    DockerExecutor,
    run_mutation_in_docker,
    sync_command,
)
from devgates.mutation import CommandOutput
from devgates.report import GateResult
from devgates.runner import Gate, run_gates

OK = CommandOutput(returncode=0, stdout="")
UNAVAILABLE = CommandOutput(returncode=1, stdout="Cannot connect to the Docker daemon")
NAME = "devgates-mutation-x"
INSPECT = ("inspect", "-f", "{{.State.Running}}", NAME)
RUN = (
    "run",
    "-d",
    "--name",
    NAME,
    "-v",
    "C:/x:/src:ro",
    "python:3.12-slim",
    "sleep",
    "infinity",
)


def sh(command: str) -> tuple[str, ...]:
    return ("exec", NAME, "sh", "-c", command)


@dataclass
class FakeDocker:
    """Faux client docker : état du conteneur et empreinte mémorisée, appels enregistrés."""

    failing: str | None = None
    running: bool = False
    stopped: bool = False
    stored_hash: str | None = None
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def __call__(self, args: Sequence[str]) -> CommandOutput:
        self.calls.append(tuple(args))
        if args[0] == self.failing:
            return UNAVAILABLE
        if args[0] == "inspect" and self.stopped:
            return CommandOutput(0, "false\n")
        if args[0] == "inspect":
            return (
                CommandOutput(0, "true\n") if self.running else CommandOutput(1, "No such object")
            )
        if tuple(args) == sh(READ_HASH):
            if self.stored_hash is None:
                return CommandOutput(1, "")
            return CommandOutput(0, self.stored_hash + "\n")
        return OK


def executor(docker: FakeDocker, deps_hash: str = "h1") -> DockerExecutor:
    return DockerExecutor(docker, name=NAME, source="C:/x", deps_hash=deps_hash)


def test_mutation_reports_status_error_when_docker_is_unavailable() -> None:
    # Arrange
    docker = FakeDocker(failing="run")

    # Act
    result = run_mutation_in_docker(executor(docker), targets=None)

    # Assert
    assert result.gate == "mutation"
    assert result.status == "error"


def test_mutation_error_finding_carries_the_docker_message() -> None:
    # Arrange
    docker = FakeDocker(failing="run")

    # Act
    result = run_mutation_in_docker(executor(docker), targets=None)

    # Assert
    assert result.error == "docker run a échoué (code 1) : Cannot connect to the Docker daemon"


def test_other_gates_still_run_when_docker_fails() -> None:
    # Arrange
    docker = FakeDocker(failing="run")
    gates = [
        Gate(name="mutation", run=lambda: run_mutation_in_docker(executor(docker), targets=None)),
        Gate(name="lint", run=lambda: GateResult(gate="lint", status="passed")),
    ]

    # Act
    report = run_gates(gates)

    # Assert
    assert [(r.gate, r.status) for r in report.gates] == [("mutation", "error"), ("lint", "passed")]


def test_docker_executor_creates_and_prepares_the_container_when_it_is_not_running() -> None:
    # Arrange
    docker = FakeDocker(running=False)

    # Act
    with executor(docker, deps_hash="h1"):
        pass

    # Assert
    assert docker.calls == [
        INSPECT,
        ("rm", "-f", NAME),
        RUN,
        sh(INSTALL),
        sh(REFRESH),
        sh(READ_HASH),
        sh(sync_command("h1")),
    ]


def test_docker_executor_reuses_a_running_container_without_recreating_it() -> None:
    # Arrange
    docker = FakeDocker(running=True, stored_hash="h1")

    # Act
    with executor(docker, deps_hash="h1"):
        pass

    # Assert
    assert docker.calls == [INSPECT, sh(REFRESH), sh(READ_HASH)]


@pytest.mark.parametrize(
    ("stored", "synced"),
    [
        pytest.param("h1", False, id="empreinte-identique"),
        pytest.param("ancienne", True, id="empreinte-differente"),
    ],
)
def test_docker_executor_installs_dependencies_only_when_the_lock_hash_changed(
    stored: str, synced: bool
) -> None:
    # Arrange
    docker = FakeDocker(running=True, stored_hash=stored)

    # Act
    with executor(docker, deps_hash="h1"):
        pass

    # Assert
    assert (sh(sync_command("h1")) in docker.calls) == synced


def test_docker_executor_refreshes_the_copy_of_the_sources_at_every_run() -> None:
    # Arrange
    docker = FakeDocker(running=True, stored_hash="h1")

    # Act
    with executor(docker):
        pass
    with executor(docker):
        pass

    # Assert
    assert docker.calls.count(sh(REFRESH)) == 2


@pytest.mark.parametrize("fails", [False, True], ids=["run-normal", "run-qui-plante"])
def test_docker_executor_leaves_the_container_running_after_the_run(fails: bool) -> None:
    # Arrange
    docker = FakeDocker(running=True, stored_hash="h1")

    # Act
    try:
        with executor(docker):
            if fails:
                raise RuntimeError("boom")
    except RuntimeError:
        pass

    # Assert
    assert not any(call[0] == "rm" for call in docker.calls)


def test_docker_executor_runs_commands_with_the_container_python_in_the_work_folder() -> None:
    # Arrange
    docker = FakeDocker(running=True, stored_hash="h1")

    # Act
    with executor(docker) as execute:
        output = execute(("mutmut", "results"))

    # Assert
    assert docker.calls[-1] == (
        "exec",
        "-w",
        "/work",
        NAME,
        "/tmp/venv/bin/python",
        "-m",
        "mutmut",
        "results",
    )
    assert output == OK


def test_docker_executor_reports_the_failing_setup_step() -> None:
    # Arrange
    docker = FakeDocker(failing="exec")

    # Act
    result = run_mutation_in_docker(executor(docker), targets=None)

    # Assert
    assert result.error == "docker exec a échoué (code 1) : Cannot connect to the Docker daemon"


def test_docker_executor_still_removes_a_container_whose_preparation_failed() -> None:
    # Arrange
    docker = FakeDocker(failing="exec")

    # Act
    run_mutation_in_docker(executor(docker), targets=None)

    # Assert
    assert docker.calls[-1] == ("rm", "-f", NAME)


def test_mutation_in_docker_returns_the_mutation_result_when_docker_works() -> None:
    # Arrange
    docker = FakeDocker()

    # Act
    result = run_mutation_in_docker(executor(docker), targets=None)

    # Assert
    assert result == GateResult(gate="mutation", status="passed")


def test_run_mutation_in_docker_does_not_touch_docker_when_there_are_no_targets() -> None:
    # Arrange
    docker = FakeDocker()

    # Act
    result = run_mutation_in_docker(executor(docker), targets=())

    # Assert
    assert (result, docker.calls) == (GateResult(gate="mutation", status="passed"), [])


def test_docker_executor_recreates_a_container_that_exists_but_is_stopped() -> None:
    # Arrange
    docker = FakeDocker(stopped=True)

    # Act
    with executor(docker):
        pass

    # Assert
    assert docker.calls[:3] == [INSPECT, ("rm", "-f", NAME), RUN]


def test_sync_command_installs_in_the_container_venv_and_stores_the_hash() -> None:
    # Arrange / Act
    command = sync_command("abc123")

    # Assert
    assert command == (
        "cd /work && UV_PROJECT_ENVIRONMENT=/tmp/venv uv sync -q && echo abc123 > /tmp/deps.sha256"
    )


def test_run_mutation_in_docker_passes_the_targets_to_mutmut() -> None:
    # Arrange
    docker = FakeDocker(running=True, stored_hash="h1")

    # Act
    run_mutation_in_docker(executor(docker), targets=("devgates.docker.*",))

    # Assert
    expected = (
        "exec",
        "-w",
        "/work",
        NAME,
        "/tmp/venv/bin/python",
        "-m",
        "mutmut",
        "run",
        "devgates.docker.*",
    )
    assert expected in docker.calls
