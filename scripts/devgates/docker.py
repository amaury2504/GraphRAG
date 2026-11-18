"""Conteneur Linux persistant pour la mutation (mutmut ne tourne pas nativement sous Windows).
Un échec de Docker devient un gate en état `error` avec le message de Docker, jamais un plantage."""

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from types import TracebackType

from devgates.mutation import CommandOutput, Executor, run_mutation
from devgates.report import GateResult

DockerCli = Callable[[Sequence[str]], CommandOutput]


class DockerError(Exception):
    """Une commande docker a échoué."""


# `git` : les tests de la PR de feature du produit (A-83) lancent un vrai dépôt git.
INSTALL = "apt-get update -qq && apt-get install -y -qq git && pip install -q uv"
# Sources fraîches à chaque run, sans les dossiers volumineux ou propres à la machine hôte.
REFRESH = (
    "rm -rf /work && mkdir /work && tar -C /src --exclude=./.venv --exclude=./.git "
    "--exclude=./mutants --exclude=./reports --exclude=__pycache__ -cf - . | tar -C /work -xf -"
)
READ_HASH = "cat /tmp/deps.sha256"


def sync_command(deps_hash: str) -> str:
    return (
        "cd /work && UV_PROJECT_ENVIRONMENT=/tmp/venv uv sync -q && "
        f"echo {deps_hash} > /tmp/deps.sha256"
    )


class DockerExecutor:
    """Reprend le conteneur du dossier s'il tourne ; sinon le crée. Les sources sont recopiées à
    chaque run, les dépendances ne sont réinstallées que si leur empreinte change."""

    def __init__(self, docker: DockerCli, name: str, source: str, deps_hash: str) -> None:
        self._docker = docker
        self._name = name
        self._source = source
        self._deps_hash = deps_hash

    def __enter__(self) -> Executor:
        try:
            if not self._running():
                self._create()
            self._sh(REFRESH)
            stored = self._docker(("exec", self._name, "sh", "-c", READ_HASH))
            if stored.returncode != 0 or stored.stdout.strip() != self._deps_hash:
                self._sh(sync_command(self._deps_hash))
        except DockerError:
            self._docker(("rm", "-f", self._name))  # __exit__ n'est pas appelé si __enter__ échoue
            raise
        return self._execute

    def _running(self) -> bool:
        state = self._docker(("inspect", "-f", "{{.State.Running}}", self._name))
        return state.returncode == 0 and state.stdout.strip() == "true"

    def _create(self) -> None:
        self._docker(("rm", "-f", self._name))
        self._check(
            ("run", "-d", "--name", self._name, "-v", f"{self._source}:/src:ro")
            + ("python:3.12-slim", "sleep", "infinity")
        )
        self._sh(INSTALL)

    def _sh(self, command: str) -> None:
        self._check(("exec", self._name, "sh", "-c", command))

    def _execute(self, command: Sequence[str]) -> CommandOutput:
        python = "/tmp/venv/bin/python"
        return self._docker(("exec", "-w", "/work", self._name, python, "-m", *command))

    def _check(self, args: Sequence[str]) -> None:
        output = self._docker(args)
        if output.returncode != 0:
            raise DockerError(
                f"docker {args[0]} a échoué (code {output.returncode}) : {output.stdout}"
            )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Le conteneur reste en vie pour le run suivant (`close-worktree` le retire)."""


def run_mutation_in_docker(
    session: AbstractContextManager[Executor], targets: Sequence[str] | None
) -> GateResult:
    if targets is not None and not targets:
        return GateResult(gate="mutation", status="passed")  # rien à muter : pas de Docker
    try:
        with session as execute:
            return run_mutation(execute, targets)
    except DockerError as exc:
        return GateResult(gate="mutation", status="error", error=str(exc))
