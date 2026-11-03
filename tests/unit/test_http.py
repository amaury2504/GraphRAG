import httpx
import pytest

from graphrag.adapters.http import MinIntervalLimiter, RequestFailedError, get_with_retry


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def client_with(*steps: httpx.Response | Exception) -> tuple[httpx.Client, list[httpx.Request]]:
    queue = list(steps)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        step = queue.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def test_limiter_enforces_minimum_interval() -> None:
    clock = FakeClock()
    limiter = MinIntervalLimiter(3.0, clock.time, clock.sleep)
    limiter.wait()  # first call never waits
    clock.now += 1.0
    limiter.wait()
    assert clock.sleeps == [2.0]
    clock.now += 10.0  # already longer than the interval: no wait
    limiter.wait()
    assert clock.sleeps == [2.0]


def test_returns_first_success_without_sleeping() -> None:
    client, seen = client_with(httpx.Response(200, text="ok"))
    clock = FakeClock()
    response = get_with_retry(client, "https://x.test/a", {"q": 1}, sleep=clock.sleep)
    assert response.text == "ok"
    assert len(seen) == 1
    assert clock.sleeps == []


def test_retries_on_5xx_with_exponential_backoff() -> None:
    client, seen = client_with(
        httpx.Response(503), httpx.Response(500), httpx.Response(200, text="ok")
    )
    clock = FakeClock()
    response = get_with_retry(client, "https://x.test/a", {}, sleep=clock.sleep, base_delay=2.0)
    assert response.text == "ok"
    assert len(seen) == 3
    assert clock.sleeps == [2.0, 4.0]


def test_retry_after_header_is_honoured() -> None:
    client, _ = client_with(
        httpx.Response(429, headers={"Retry-After": "30"}), httpx.Response(200, text="ok")
    )
    clock = FakeClock()
    get_with_retry(client, "https://x.test/a", {}, sleep=clock.sleep, base_delay=2.0)
    assert clock.sleeps == [30.0]


def test_retries_on_network_error() -> None:
    client, seen = client_with(httpx.ConnectError("down"), httpx.Response(200, text="ok"))
    clock = FakeClock()
    assert get_with_retry(client, "https://x.test/a", {}, sleep=clock.sleep).text == "ok"
    assert len(seen) == 2


def test_gives_up_after_max_attempts() -> None:
    client, seen = client_with(*[httpx.Response(503)] * 3)
    clock = FakeClock()
    with pytest.raises(RequestFailedError, match="3 attempts"):
        get_with_retry(client, "https://x.test/a", {}, sleep=clock.sleep, max_attempts=3)
    assert len(seen) == 3
    assert len(clock.sleeps) == 2  # no pointless sleep after the last attempt


def test_client_error_is_not_retried() -> None:
    client, seen = client_with(httpx.Response(404))
    with pytest.raises(RequestFailedError, match="HTTP 404"):
        get_with_retry(client, "https://x.test/a", {}, sleep=FakeClock().sleep)
    assert len(seen) == 1


def test_error_body_in_200_is_retried() -> None:
    client, seen = client_with(httpx.Response(200, text="BAD"), httpx.Response(200, text="good"))
    response = get_with_retry(
        client,
        "https://x.test/a",
        {},
        sleep=FakeClock().sleep,
        is_error_body=lambda r: r.text == "BAD",
    )
    assert response.text == "good"
    assert len(seen) == 2


def test_limiter_is_applied_to_every_attempt() -> None:
    client, _ = client_with(httpx.Response(503), httpx.Response(200, text="ok"))
    clock = FakeClock()
    limiter = MinIntervalLimiter(3.0, clock.time, clock.sleep)
    get_with_retry(
        client, "https://x.test/a", {}, limiter=limiter, sleep=clock.sleep, base_delay=1.0
    )
    # attempt 1: no wait; backoff 1s; attempt 2: limiter tops up to 3s total
    assert clock.sleeps == [1.0, 2.0]
