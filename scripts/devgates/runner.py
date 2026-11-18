"""Exécute des gates dans l'ordre, sans s'arrêter au premier échec, et agrège le rapport."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from devgates.report import GateResult, Report


class UnknownGateError(Exception):
    """Le nom de gate demandé n'existe pas."""


class InvalidGateSelectionError(Exception):
    """Sélection invalide : un rapport vide serait un faux succès."""


@dataclass(frozen=True)
class Gate:
    name: str
    run: Callable[[], GateResult]


def run_gates(gates: Sequence[Gate], only: str | None = None, skip: Sequence[str] = ()) -> Report:
    if only is not None and skip:
        raise InvalidGateSelectionError("--only et --skip ensemble : choisir l'un ou l'autre")
    selected = list(gates)
    if only is not None:
        selected = [gate for gate in gates if gate.name == only]
        if not selected:
            raise UnknownGateError(f"gate inconnu : {only}")
    for name in skip:
        if all(gate.name != name for gate in gates):
            raise UnknownGateError(f"gate inconnu : {name}")
    selected = [gate for gate in selected if gate.name not in skip]
    if not selected:
        raise InvalidGateSelectionError("aucun gate à exécuter : tous sont ignorés")
    return Report(gates=tuple(_run_one(gate) for gate in selected))


def _run_one(gate: Gate) -> GateResult:
    try:
        return gate.run()
    except Exception as exc:  # un outil défaillant est un état `error`, pas un plantage
        return GateResult(gate=gate.name, status="error", error=str(exc))
