"""Rapport JSON des gates (schéma versionné, lisible par des agents)."""

import json
from dataclasses import dataclass
from typing import Literal, TypedDict

SCHEMA_VERSION = 1

GateStatus = Literal["passed", "failed", "error"]


class FindingDict(TypedDict):
    cause: str
    file: str | None
    line: int | None
    rule: str | None
    mutant: str | None
    diff: str | None


class GateDict(TypedDict):
    gate: str
    status: GateStatus
    error: str | None
    findings: list[FindingDict]


class ReportDict(TypedDict):
    schema_version: int
    status: GateStatus
    exit_code: int
    gates: list[GateDict]


@dataclass(frozen=True)
class Finding:
    cause: str
    file: str | None = None
    line: int | None = None
    rule: str | None = None
    mutant: str | None = None
    diff: str | None = None


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: GateStatus
    findings: tuple[Finding, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class Report:
    gates: tuple[GateResult, ...]

    @property
    def status(self) -> GateStatus:
        statuses = {result.status for result in self.gates}
        if "error" in statuses:
            return "error"
        if "failed" in statuses:
            return "failed"
        return "passed"

    @property
    def exit_code(self) -> int:
        return 0 if self.status == "passed" else 1

    def to_dict(self) -> ReportDict:
        return ReportDict(
            schema_version=SCHEMA_VERSION,
            status=self.status,
            exit_code=self.exit_code,
            gates=[
                GateDict(
                    gate=result.gate,
                    status=result.status,
                    error=result.error,
                    findings=[
                        FindingDict(
                            cause=finding.cause,
                            file=finding.file,
                            line=finding.line,
                            rule=finding.rule,
                            mutant=finding.mutant,
                            diff=finding.diff,
                        )
                        for finding in result.findings
                    ],
                )
                for result in self.gates
            ],
        )

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            indent=2,
            ensure_ascii=False,  # pragma: no mutate  (None se comporte comme False : équivalent)
        )
