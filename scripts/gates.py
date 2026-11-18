"""Point d'entrée des gates : appelé par le hook de commit ET par la CI (source unique).

Usage : python scripts/gates.py [--only NOM | --skip NOM ...] [--mutation changed|full]

La logique (rapport, runner, gate mutation) vit dans `devgates`. Ce fichier ne fait que le
câblage : catalogue des gates et exécuteurs de commandes réels. Les gates lint/format/types/tests
lisent la sortie JSON/XML de leur outil (`devgates.parsers`) pour produire un finding par
problème, avec fichier et ligne (voir docs/journal-architecture.md, A-22).
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path

from devgates.bypass import run_bypass
from devgates.container import container_name
from devgates.docker import DockerExecutor, run_mutation_in_docker
from devgates.lock import RunLockedError, run_lock
from devgates.mutation import CommandOutput, run_mutation, select_environment
from devgates.parsers import parse_coverage, parse_junit, parse_mypy, parse_ruff
from devgates.report import Finding, GateResult
from devgates.runner import Gate, InvalidGateSelectionError, UnknownGateError, run_gates
from devgates.summary import summarize
from devgates.targets import mutation_targets
from devgates.tool_gate import Parser, run_tool_gate

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "reports" / "gates.json"
PY = [sys.executable, "-m"]
LOCK_PATH = ROOT / "reports" / "gates.lock"
JUNIT_PATH = ROOT / "reports" / "junit.xml"
COVERAGE_PATH = ROOT / "reports" / "coverage.json"
ruff_parser = partial(parse_ruff, root=str(ROOT))
mypy_parser = partial(parse_mypy, root=str(ROOT))
BASE = "origin/main"  # branche de référence de la mutation ciblée


def tool_gate(name: str, command: Sequence[str], parse: Parser) -> Gate:
    def run() -> GateResult:
        done = subprocess.run(
            [*PY, *command],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        # stdout seul : mypy et ruff y écrivent leur JSON ; stderr ne sert que si stdout est vide
        output = CommandOutput(returncode=done.returncode, stdout=done.stdout or done.stderr)
        return run_tool_gate(name, output, parse)

    return Gate(name=name, run=run)


def command_gate(name: str, command: Sequence[str], cwd: Path = ROOT) -> Gate:
    """Gate sur un exécutable externe (hors `python -m`) : code de sortie et fin de sortie."""

    def run() -> GateResult:
        try:
            done = subprocess.run(
                list(command),
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except OSError as exc:  # exécutable introuvable
            return run_tool_gate(name, CommandOutput(returncode=127, stdout=str(exc)), _no_finding)
        output = CommandOutput(returncode=done.returncode, stdout=done.stdout + done.stderr)
        return run_tool_gate(name, output, _no_finding)

    return Gate(name=name, run=run)


def _no_finding(_stdout: str) -> tuple[Finding, ...]:
    """Sans format structuré : `run_tool_gate` retombe sur le code de sortie et la fin de sortie."""
    return ()


def parse_test_reports(_stdout: str) -> tuple[Finding, ...]:
    """Tests en échec (JUnit) puis couverture manquante (coverage json), s'ils ont été écrits."""
    findings: tuple[Finding, ...] = ()
    if JUNIT_PATH.exists():
        findings += parse_junit(JUNIT_PATH.read_text(encoding="utf-8"))
    if COVERAGE_PATH.exists():
        findings += parse_coverage(COVERAGE_PATH.read_text(encoding="utf-8"))
    return findings


def tests_gate() -> Gate:
    command = [
        "pytest",
        "--cov",  # lignes + branches, fail_under=100
        "--cov-report=term",
        f"--cov-report=json:{COVERAGE_PATH}",
        f"--junitxml={JUNIT_PATH}",
        "-o",
        "junit_family=xunit1",
    ]
    gate = tool_gate("tests", command, parse_test_reports)

    def run() -> GateResult:
        for stale in (JUNIT_PATH, COVERAGE_PATH):  # un rapport d'un lancement précédent mentirait
            stale.unlink(missing_ok=True)
        REPORT_PATH.parent.mkdir(exist_ok=True)
        return gate.run()

    return Gate(name="tests", run=run)


_PROGRESS = re.compile(rb"(\d+)/(\d+)")
_BAR_WIDTH = 30


def run_with_progress(command: Sequence[str], cwd: Path | None) -> CommandOutput:
    """Lance une commande de mutation en montrant sa progression sur stderr (stdout reste le JSON).

    `mutmut` écrit son compteur `fait/total` en continu ; la sortie complète est gardée, comme avec
    `capture_output`, pour le message d'erreur du gate.
    """
    process = subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
    )
    assert process.stdout is not None
    captured = bytearray()
    while chunk := os.read(process.stdout.fileno(), 4096):
        captured += chunk
        counters = _PROGRESS.findall(chunk)
        if counters:
            done, total = (int(number) for number in counters[-1])
            filled = _BAR_WIDTH * done // total if total else 0
            bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
            percent = 100 * done // total if total else 0
            sys.stderr.write(f"\rmutation [{bar}] {done}/{total} ({percent} %)")
            sys.stderr.flush()
    process.wait()
    sys.stderr.write("\n")
    return CommandOutput(
        returncode=process.returncode, stdout=captured.decode("utf-8", errors="replace")
    )


def native_execute(command: Sequence[str]) -> CommandOutput:
    if "mutmut" in command:
        return run_with_progress([*PY, *command], ROOT)
    done = subprocess.run(
        [*PY, *command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return CommandOutput(returncode=done.returncode, stdout=done.stdout + done.stderr)


def docker_cli(args: Sequence[str]) -> CommandOutput:
    try:
        if "mutmut" in args:
            return run_with_progress(["docker", *args], None)
        done = subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:  # docker introuvable
        return CommandOutput(returncode=127, stdout=str(exc))
    return CommandOutput(returncode=done.returncode, stdout=done.stdout + done.stderr)


def process_alive(pid: int) -> bool:
    """`os.kill(pid, 0)` TUE le processus sous Windows : on interroge le système à la place."""
    on_windows = sys.platform == "win32"  # variable : mypy ne coupe pas la branche POSIX
    if on_windows:
        listing = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        return f'"{pid}"' in listing
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def git_execute(command: Sequence[str]) -> CommandOutput:
    done = subprocess.run(
        list(command),
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return CommandOutput(returncode=done.returncode, stdout=done.stdout + done.stderr)


def deps_hash() -> str:
    """Empreinte des dépendances : change seulement quand `uv sync` doit être refait."""
    digest = hashlib.sha256()
    for name in ("pyproject.toml", "uv.lock"):
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def mutation_gate(mode: str, explicit: Sequence[str] = ()) -> GateResult:
    """`changed` : mutants des seuls fichiers de la feature (repli sur `full` si git échoue).
    `explicit` : motifs de mutants donnés à la main, qui remplacent la sélection par git."""
    if explicit:
        targets: tuple[str, ...] | None = tuple(explicit)
    else:
        targets = mutation_targets(git_execute, base=BASE) if mode == "changed" else None
    if select_environment(sys.platform) == "docker":
        session = DockerExecutor(docker_cli, container_name(ROOT.name), str(ROOT), deps_hash())
        return run_mutation_in_docker(session, targets)
    return run_mutation(native_execute, targets)


def bypass_gate() -> GateResult:
    sources = sorted([*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").glob("*.py")])
    return run_bypass(sources)


def build_gates(mutation_mode: str, explicit: Sequence[str] = ()) -> list[Gate]:
    return [
        tool_gate("lint", ["ruff", "check", "--output-format", "json", "."], ruff_parser),
        tool_gate(
            "format", ["ruff", "format", "--check", "--output-format", "json", "."], ruff_parser
        ),
        tool_gate("types", ["mypy", "--output", "json"], mypy_parser),
        tests_gate(),
        command_gate("gitleaks", ["gitleaks", "git", "--pre-commit", "--staged", "--redact"]),
        Gate(name="bypass", run=bypass_gate),
        Gate(name="mutation", run=lambda: mutation_gate(mutation_mode, explicit)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    names = [gate.name for gate in build_gates("changed")]
    parser.add_argument("--only", choices=names)
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=names,
        help="gate à ignorer (répétable) ; le hook et la CI ignorent la mutation (A-37)",
    )
    parser.add_argument(
        "--mutation",
        choices=["changed", "full"],
        default="changed",
        help="changed : mutants des fichiers de la feature ; full : tout (CI de nuit)",
    )
    parser.add_argument(
        "--mutation-target",
        action="append",
        default=[],
        help="motif de mutants (répétable), par exemple app.module.* : "
        "remplace la sélection par git, pour revérifier un fichier sans refaire les autres",
    )
    args = parser.parse_args()
    try:
        with run_lock(LOCK_PATH, os.getpid(), process_alive):
            report = run_gates(
                build_gates(args.mutation, args.mutation_target), only=args.only, skip=args.skip
            )
    except (UnknownGateError, InvalidGateSelectionError, RunLockedError) as exc:
        parser.error(str(exc))
    document = report.to_json()
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(document, encoding="utf-8")
    # Octets UTF-8 : la console Windows (cp1252) ne sait pas afficher les noms de mutants (ǁ).
    sys.stdout.buffer.write((document + "\n").encode("utf-8"))
    summary = summarize(report)
    if summary:  # résumé lisible pour l'humain (hook de commit) ; le JSON reste pour les agents
        sys.stderr.buffer.write((summary + "\n").encode("utf-8"))
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
