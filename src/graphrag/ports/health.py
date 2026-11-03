from typing import Protocol


class HealthProbe(Protocol):
    """Checks that one backing service is reachable."""

    name: str

    def check(self) -> None:
        """Return normally if the service is healthy, raise any exception otherwise."""
        ...
