"""Résumé court du rapport pour la console (le JSON reste la sortie des agents)."""

from devgates.report import Finding, Report

SHOWN = 3  # findings affichés par gate


def summarize(report: Report) -> str:
    lines: list[str] = []
    for result in report.gates:
        if result.status == "passed":
            continue
        lines.append(f"{result.gate} : {result.status}")
        if result.error is not None:
            lines.append(f"  {result.error}")
        lines.extend(f"  {_describe(finding)}" for finding in result.findings[:SHOWN])
        if len(result.findings) > SHOWN:
            lines.append(f"  … et {len(result.findings) - SHOWN} autres")
    return "\n".join(lines)


def _describe(finding: Finding) -> str:
    parts: list[str] = []
    if finding.file is not None:
        parts.append(finding.file if finding.line is None else f"{finding.file}:{finding.line}")
    if finding.rule is not None:
        parts.append(finding.rule)
    parts.append(finding.cause)
    return " ".join(parts)
