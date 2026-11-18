"""Gate d'un outil externe : code de sortie + analyse de sa sortie en findings."""

from collections.abc import Callable

from devgates.mutation import CommandOutput
from devgates.parsers import ParseError
from devgates.report import Finding, GateResult

TAIL_LINES = 10  # fin de la sortie gardée quand rien n'est analysable (ex. couverture < 100 %)

Parser = Callable[[str], tuple[Finding, ...]]


def run_tool_gate(name: str, output: CommandOutput, parse: Parser) -> GateResult:
    if output.returncode == 0:
        return GateResult(gate=name, status="passed")
    try:
        findings = parse(output.stdout)
    except ParseError as exc:
        return GateResult(gate=name, status="error", error=str(exc))
    if not findings:
        tail = "\n".join(output.stdout.splitlines()[-TAIL_LINES:])
        findings = (Finding(cause=f"code de sortie {output.returncode} : {tail}"),)
    return GateResult(gate=name, status="failed", findings=findings)
