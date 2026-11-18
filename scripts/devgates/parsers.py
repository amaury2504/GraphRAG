"""Analyseurs de la sortie des outils (ruff, mypy, pytest, coverage) : texte → findings.
Fonctions pures : le gate décide quoi faire d'une sortie illisible (`ParseError` → état `error`)."""

import json
import xml.etree.ElementTree as ET

from devgates.report import Finding


class ParseError(Exception):
    """La sortie d'un outil n'a pas la forme attendue."""


def _slashes(path: str) -> str:
    """Séparateurs `/` : la sortie des outils dépend de la plateforme."""
    return path.replace("\\", "/")


def _relative(path: str, root: str) -> str:
    return _slashes(path).removeprefix(_slashes(root) + "/")


def parse_ruff(text: str, root: str) -> tuple[Finding, ...]:
    """Sortie `--output-format json` de `ruff check` et de `ruff format --check`."""
    try:
        violations = json.loads(text)
        if not isinstance(violations, list):
            raise TypeError("liste attendue")
        return tuple(
            Finding(
                cause=violation["message"],
                file=_relative(violation["filename"], root),
                line=violation["location"]["row"],
                rule=violation["code"],
            )
            for violation in violations
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise ParseError(f"sortie de ruff illisible : {exc}") from exc


def parse_mypy(text: str, root: str) -> tuple[Finding, ...]:
    """Sortie `--output json` de mypy : un objet JSON par ligne ; les notes sont ignorées."""
    try:
        lines = [json.loads(line) for line in text.splitlines() if line.strip()]
        return tuple(
            Finding(
                cause=line["message"],
                file=_relative(line["file"], root),
                line=line["line"],
                rule=line["code"],
            )
            for line in lines
            if line["severity"] == "error"
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise ParseError(f"sortie de mypy illisible : {exc}") from exc


def parse_junit(text: str) -> tuple[Finding, ...]:
    """Rapport JUnit (`xunit1`) de pytest : un finding par test en échec ou en erreur."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ParseError(f"sortie de pytest illisible : {exc}") from exc
    findings = []
    for case in root.findall(".//testcase"):
        problem = case.find("failure")
        if problem is None:
            problem = case.find("error")
        if problem is not None:
            message = problem.get("message", "")
            file = case.get("file")
            findings.append(
                Finding(
                    cause=f"{case.get('name')} : {message}",
                    file=None if file is None else _slashes(file),
                )
            )
    return tuple(findings)


def parse_coverage(text: str) -> tuple[Finding, ...]:
    """`coverage json` : un finding par fichier ayant des lignes ou des branches non couvertes."""
    try:
        files = json.loads(text)["files"]
        return tuple(
            Finding(cause=cause, file=_slashes(name))
            for name, data in files.items()
            if (cause := _uncovered(data["missing_lines"], data["missing_branches"]))
        )
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ParseError(f"sortie de coverage illisible : {exc}") from exc


def _uncovered(missing_lines: list[int], missing_branches: list[list[int]]) -> str:
    parts = []
    if missing_lines:
        parts.append("lignes non couvertes : " + ", ".join(str(n) for n in missing_lines))
    if missing_branches:
        branches = (f"{start}→{'sortie' if end < 0 else end}" for start, end in missing_branches)
        parts.append("branches non couvertes : " + ", ".join(branches))
    return " ; ".join(parts)
