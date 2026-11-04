import pytest

from graphrag.adapters.token_limiter import TokenRateLimiter


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def limiter(clock: Clock, limit: int = 100) -> TokenRateLimiter:
    return TokenRateLimiter(limit, clock=clock, sleep=clock.sleep)


def test_calls_within_the_quota_do_not_wait() -> None:
    clock = Clock()
    lim = limiter(clock)
    lim.acquire(40)
    lim.acquire(60)  # exactly the quota
    assert clock.sleeps == []


def test_a_call_over_the_quota_waits_for_the_oldest_to_leave_the_window() -> None:
    clock = Clock()
    lim = limiter(clock)
    lim.acquire(60)
    clock.now = 10.0
    lim.acquire(30)
    lim.acquire(20)  # 110 > 100: wait until the first entry (t=0) expires at t=60
    assert clock.sleeps == [50.0]
    assert clock.now == 60.0


def test_settling_replaces_the_estimate_with_the_real_usage() -> None:
    clock = Clock()
    lim = limiter(clock)
    first = lim.acquire(90)
    lim.settle(first, 10)  # the real call used far less than estimated
    lim.acquire(80)  # 10 + 80 fits: no wait
    assert clock.sleeps == []


def test_settling_with_more_than_estimated_makes_the_next_call_wait() -> None:
    clock = Clock()
    lim = limiter(clock)
    first = lim.acquire(10)
    lim.settle(first, 95)
    lim.acquire(10)
    assert clock.sleeps == [60.0]


def test_entries_older_than_the_window_are_forgotten() -> None:
    clock = Clock()
    lim = limiter(clock)
    lim.acquire(100)
    clock.now = 60.0  # the window is [t - 60, t): the entry at 0 is out
    lim.acquire(100)
    assert clock.sleeps == []


def test_a_request_larger_than_the_quota_goes_through_once_the_window_is_empty() -> None:
    clock = Clock()
    lim = limiter(clock, limit=100)
    lim.acquire(50)
    lim.acquire(500)  # can never fit: waits for the window to empty, then proceeds
    assert clock.sleeps == [60.0]
    lim.acquire(10)  # but it fills the window for the next call
    assert clock.sleeps == [60.0, 60.0]


def test_an_empty_window_never_blocks_even_an_oversized_first_call() -> None:
    clock = Clock()
    limiter(clock, limit=10).acquire(1000)
    assert clock.sleeps == []


def test_the_quota_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        TokenRateLimiter(0)
