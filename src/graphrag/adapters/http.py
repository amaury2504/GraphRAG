import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class RequestFailedError(Exception):
    """A request kept failing after all retries."""


class MinIntervalLimiter:
    """Guarantees at least `interval` seconds between two calls to `wait`."""

    def __init__(
        self,
        interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            remaining = self._interval - (self._clock() - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()


def get_with_retry(
    client: httpx.Client, url: str, params: Mapping[str, str | int], **kwargs: Any
) -> httpx.Response:
    return send_with_retry(client, "GET", url, params, **kwargs)


def send_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    params: Mapping[str, str | int],
    *,
    json: object = None,
    limiter: MinIntervalLimiter | None = None,
    max_attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 600.0,
    sleep: Callable[[float], None] = time.sleep,
    is_error_body: Callable[[httpx.Response], bool] = lambda _r: False,
) -> httpx.Response:
    """Request with exponential backoff on network errors, 429/5xx and soft errors in a 200."""
    last_problem = "no attempt made"
    for attempt in range(max_attempts):
        if limiter is not None:
            limiter.wait()
        delay = min(base_delay * 2**attempt, max_delay)
        try:
            response = client.request(method, url, params=params, json=json)
        except httpx.TransportError as exc:
            last_problem = f"network error: {type(exc).__name__}"
        else:
            if response.status_code in RETRYABLE_STATUS:
                last_problem = f"HTTP {response.status_code}"
                delay = max(delay, _retry_after(response))
            elif response.status_code >= 400:
                raise RequestFailedError(f"HTTP {response.status_code} for {url}")
            elif is_error_body(response):
                last_problem = "server reported an error in a 200 response"
            else:
                return response
        if attempt < max_attempts - 1:
            logger.warning(
                "%s on %s, attempt %d/%d, retrying in %.0fs",
                last_problem,
                url,
                attempt + 1,
                max_attempts,
                delay,
            )
            sleep(delay)
    raise RequestFailedError(f"{url} failed after {max_attempts} attempts ({last_problem})")


def _retry_after(response: httpx.Response) -> float:
    try:
        return float(response.headers.get("Retry-After", 0))
    except ValueError:
        return 0.0
