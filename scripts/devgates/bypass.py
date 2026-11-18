"""Gate anti-contournement : refuse les moyens de passer outre le typage strict (R-4).

Interdits dans le code source : `model_construct`, `model_copy(update=...)`, `typing.cast` et
`# type: ignore` sans code d'erreur. Analyse par `ast` (appels) et `tokenize` (commentaires) :
les noms cités dans une chaîne ou un commentaire ne comptent pas.
"""

import ast
import io
import re
import tokenize
from collections.abc import Sequence
from pathlib import Path

from devgates.report import Finding, GateResult

_IGNORE_WITHOUT_CODE = re.compile(r"type:\s*ignore(?!\[)")

_MODEL_CONSTRUCT = ("BYPASS-MODEL-CONSTRUCT", "model_construct contourne la validation")
_MODEL_COPY_UPDATE = ("BYPASS-MODEL-COPY-UPDATE", "model_copy(update=...) contourne la validation")
_CAST = ("BYPASS-CAST", "typing.cast contourne le typage")
_TYPE_IGNORE = ("BYPASS-TYPE-IGNORE", "type: ignore sans code d'erreur")


def _call_rule(call: ast.Call) -> tuple[str, str] | None:
    name = ast.unparse(call.func).split(".")[-1]
    if name == "model_construct":
        return _MODEL_CONSTRUCT
    if name == "model_copy" and any(keyword.arg == "update" for keyword in call.keywords):
        return _MODEL_COPY_UPDATE
    if name == "cast":
        return _CAST
    return None


def _findings_in(path: Path) -> list[Finding]:
    source = path.read_bytes()  # l'encodage est lu par Python (PEP 263), pas deviné ici
    found: list[tuple[int, tuple[str, str]]] = []
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if isinstance(node, ast.Call) and (rule := _call_rule(node)) is not None:
            found.append((node.lineno, rule))
    for token in tokenize.tokenize(io.BytesIO(source).readline):
        if token.type == tokenize.COMMENT and _IGNORE_WITHOUT_CODE.search(token.string):
            found.append((token.start[0], _TYPE_IGNORE))
    return [
        Finding(cause=cause, file=str(path), line=line, rule=rule)
        for line, (rule, cause) in sorted(found)
    ]


def run_bypass(paths: Sequence[Path]) -> GateResult:
    findings: list[Finding] = []
    for path in paths:
        try:
            findings.extend(_findings_in(path))
        except SyntaxError as exc:
            return GateResult(gate="bypass", status="error", error=f"{path} : {exc}")
    if findings:
        return GateResult(gate="bypass", status="failed", findings=tuple(findings))
    return GateResult(gate="bypass", status="passed")
