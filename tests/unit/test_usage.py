import pytest

from graphrag.domain.models import Completion, Usage
from graphrag.pipeline.usage import MeteredLLM, UsageMeter


def test_usage_adds_up() -> None:
    total = Usage(10, 2) + Usage(5, 1)
    assert (total.prompt_tokens, total.completion_tokens, total.total_tokens) == (15, 3, 18)


def test_meter_totals_every_call() -> None:
    meter = UsageMeter()
    meter.record(Usage(100, 10))
    meter.record(Usage(50, 5))
    assert meter.total == Usage(150, 15)
    assert meter.calls == 2


def test_meter_warns_once_when_the_total_passes_the_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    meter = UsageMeter(alert_tokens=100)
    with caplog.at_level("WARNING", logger="graphrag.pipeline.usage"):
        meter.record(Usage(60, 40))  # exactly 100: not past it
        assert caplog.records == []
        meter.record(Usage(1, 0))
        meter.record(Usage(500, 50))
    assert [r.getMessage() for r in caplog.records] == [
        "language model usage passed 100 tokens (101 so far, 2 calls)"
    ]


def test_no_threshold_never_warns(caplog: pytest.LogCaptureFixture) -> None:
    meter = UsageMeter(alert_tokens=None)
    with caplog.at_level("WARNING", logger="graphrag.pipeline.usage"):
        meter.record(Usage(10**9, 10**9))
    assert caplog.records == []


class FakeLLM:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def complete(self, system: str, user: str) -> Completion:
        if self.fail:
            raise RuntimeError("down")
        return Completion("text", Usage(7, 3))


def test_metered_llm_records_and_passes_through() -> None:
    meter = UsageMeter()
    assert MeteredLLM(FakeLLM(), meter).complete("s", "u") == Completion("text", Usage(7, 3))
    assert meter.total == Usage(7, 3)


def test_failed_calls_are_not_counted() -> None:
    meter = UsageMeter()
    with pytest.raises(RuntimeError):
        MeteredLLM(FakeLLM(fail=True), meter).complete("s", "u")
    assert (meter.total, meter.calls) == (Usage(), 0)
