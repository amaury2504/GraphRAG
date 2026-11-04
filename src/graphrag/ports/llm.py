from typing import Protocol

from graphrag.domain.models import Completion


class LLM(Protocol):
    def complete(self, system: str, user: str) -> Completion:
        """Answer one prompt. Raises if the model cannot be reached after retries."""
        ...
