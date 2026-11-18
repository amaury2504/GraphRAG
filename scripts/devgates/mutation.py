"""Gate mutation : tous les mutants doivent être tués. mutmut génère les mutants ; ce module
lance mutmut, lit ses résultats et les traduit en `GateResult`."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Literal

from devgates.report import Finding, GateResult

Environment = Literal["docker", "native"]


@dataclass(frozen=True)
class CommandOutput:
    returncode: int
    stdout: str


Executor = Callable[[Sequence[str]], CommandOutput]


def select_environment(platform: str) -> Environment:
    """mutmut ne tourne pas nativement sous Windows : on passe alors par Docker."""
    return "docker" if platform == "win32" else "native"


def run_mutation(execute: Executor, targets: Sequence[str] | None = None) -> GateResult:
    """`targets` : motifs de mutants à générer ; `None` = tout ; vide = rien à muter."""
    if targets is not None and not targets:
        return GateResult(gate="mutation", status="passed")
    run = execute(("mutmut", "run", *(targets or ())))
    if run.returncode != 0:
        return _error("run", run)
    listing = execute(("mutmut", "results"))
    if listing.returncode != 0:
        return _error("results", listing)
    findings = tuple(
        _finding(execute, name, status)
        for name, status in _parse_listing(listing.stdout)
        if targets is None or any(fnmatchcase(name, pattern) for pattern in targets)
    )
    return GateResult(gate="mutation", status="failed" if findings else "passed", findings=findings)


def _error(step: str, output: CommandOutput) -> GateResult:
    message = f"mutmut {step} a échoué (code {output.returncode}) : {output.stdout}"
    return GateResult(gate="mutation", status="error", error=message)


def _parse_listing(stdout: str) -> list[tuple[str, str]]:
    entries = []
    for line in stdout.splitlines():
        name, separator, status = line.strip().partition(": ")
        if separator:
            entries.append((name, status))
    return entries


def _finding(execute: Executor, name: str, status: str) -> Finding:
    diff = execute(("mutmut", "show", name)).stdout
    return Finding(cause=f"mutant {status}", mutant=name, file=_source_file(diff), diff=diff)


def _source_file(diff: str) -> str | None:
    for line in diff.splitlines():
        if line.startswith("--- "):
            return line.removeprefix("--- ")
    return None
