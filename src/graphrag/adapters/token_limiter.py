"""Keep token consumption under a per-minute quota, waiting instead of provoking 429 errors."""

import time
from collections import deque
from collections.abc import Callable


class TokenRateLimiter:
    """Sliding window: at most `tokens_per_minute` tokens spent in any `window` seconds.

    `acquire` reserves an estimate before a call (waiting if the window is full) and returns
    a reservation; `settle` replaces the estimate with the tokens actually used.
    """

    def __init__(
        self,
        tokens_per_minute: int,
        *,
        window: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if tokens_per_minute < 1:
            raise ValueError("tokens_per_minute must be positive")
        self._limit = tokens_per_minute
        self._window = window
        self._clock = clock
        self._sleep = sleep
        self._entries: deque[list[float]] = deque()  # [time, tokens], oldest first

    def _used(self) -> float:
        return sum(tokens for _, tokens in self._entries)

    def _forget_expired(self) -> None:
        now = self._clock()
        while self._entries and now - self._entries[0][0] >= self._window:
            self._entries.popleft()

    def acquire(self, estimate: int) -> list[float]:
        """Wait until `estimate` tokens fit in the window, then reserve them.

        A single request larger than the quota waits for an empty window and then goes through:
        refusing it would block it forever.
        """
        while True:
            self._forget_expired()
            if not self._entries or self._used() + estimate <= self._limit:
                break
            self._sleep(max(self._entries[0][0] + self._window - self._clock(), 0.0))
        reservation = [self._clock(), float(estimate)]
        self._entries.append(reservation)
        return reservation

    @staticmethod
    def settle(reservation: list[float], actual: int) -> None:
        reservation[1] = float(actual)
