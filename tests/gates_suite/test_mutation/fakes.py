from collections.abc import Sequence
from dataclasses import dataclass, field

from devgates.mutation import CommandOutput


@dataclass
class FakeExecutor:
    """Faux exécuteur de commandes : renvoie des sorties scénarisées et mémorise les appels."""

    responses: dict[tuple[str, ...], CommandOutput]
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def __call__(self, command: Sequence[str]) -> CommandOutput:
        key = tuple(command)
        self.calls.append(key)
        return self.responses[key]


RUN_OK = CommandOutput(returncode=0, stdout="")
NO_SURVIVOR = CommandOutput(returncode=0, stdout="")
SURVIVOR_NAME = "app.x_version__mutmut_1"
SURVIVOR_LISTING = CommandOutput(returncode=0, stdout=f"    {SURVIVOR_NAME}: survived\n")
SURVIVOR_DIFF = (
    f"# {SURVIVOR_NAME}: survived\n"
    "--- src/app/__init__.py\n"
    "+++ src/app/__init__.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def version() -> str:\n"
    '-    return "0.0.1"\n'
    '+    return "XX0.0.1XX"\n'
)
