import logging
import threading

from graphrag.domain.models import Completion, Usage
from graphrag.ports.llm import LLM

logger = logging.getLogger(__name__)


class UsageMeter:
    """Adds up the tokens of every call and warns once when the total passes `alert_tokens`.

    It only warns: nothing is ever refused, so a long evaluation is not cut in half.
    """

    def __init__(self, alert_tokens: int | None = None) -> None:
        self._alert_tokens = alert_tokens
        self._lock = threading.Lock()
        self._total = Usage()
        self._calls = 0
        self._alerted = False

    @property
    def total(self) -> Usage:
        return self._total

    @property
    def calls(self) -> int:
        return self._calls

    def record(self, usage: Usage) -> None:
        with self._lock:
            self._total += usage
            self._calls += 1
            over = self._alert_tokens is not None and self._total.total_tokens > self._alert_tokens
            if over and not self._alerted:
                self._alerted = True
                logger.warning(
                    "language model usage passed %d tokens (%d so far, %d calls)",
                    self._alert_tokens,
                    self._total.total_tokens,
                    self._calls,
                )


class MeteredLLM:
    """An LLM that reports every call to a meter."""

    def __init__(self, inner: LLM, meter: UsageMeter) -> None:
        self._inner = inner
        self._meter = meter

    def complete(self, system: str, user: str) -> Completion:
        completion = self._inner.complete(system, user)
        self._meter.record(completion.usage)
        return completion
