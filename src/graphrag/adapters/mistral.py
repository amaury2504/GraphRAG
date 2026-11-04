"""Mistral chat completions (https://docs.mistral.ai/api/endpoint/chat)."""

import time
from collections.abc import Callable
from typing import Any

import httpx

from graphrag.adapters.http import send_with_retry
from graphrag.adapters.token_limiter import TokenRateLimiter
from graphrag.domain.models import Completion, Usage

API_URL = "https://api.mistral.ai/v1/chat/completions"
DEFAULT_MODEL = "mistral-small-latest"
# The free tier allows 100 000 tokens a minute; stay under it with some margin.
DEFAULT_TOKENS_PER_MINUTE = 80_000
CHARS_PER_TOKEN = 3  # pessimistic for English; the real usage replaces the estimate afterwards


def mistral_client(
    api_key: str, *, timeout: float = 60.0, transport: httpx.BaseTransport | None = None
) -> httpx.Client:
    """An HTTP client carrying the key; the key is never part of a URL, log or error message."""
    return httpx.Client(
        timeout=timeout, headers={"Authorization": f"Bearer {api_key}"}, transport=transport
    )


class MistralLLM:
    def __init__(
        self,
        client: httpx.Client,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 700,
        temperature: float = 0.0,
        max_attempts: int = 5,
        retry_delay: float = 2.0,
        max_delay: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        tokens_per_minute: int | None = DEFAULT_TOKENS_PER_MINUTE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_attempts = max_attempts
        self._retry_delay = retry_delay
        self._max_delay = max_delay
        self._sleep = sleep
        self._limiter = (
            None
            if tokens_per_minute is None
            else TokenRateLimiter(tokens_per_minute, clock=clock, sleep=sleep)
        )

    def complete(self, system: str, user: str) -> Completion:
        reservation = None
        if self._limiter is not None:
            estimate = (len(system) + len(user)) // CHARS_PER_TOKEN + self._max_tokens
            reservation = self._limiter.acquire(estimate)
        response = send_with_retry(
            self._client,
            "POST",
            API_URL,
            {},
            json={
                "model": self._model,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            max_attempts=self._max_attempts,
            base_delay=self._retry_delay,
            max_delay=self._max_delay,
            sleep=self._sleep,
        )
        completion = parse_completion(response.json())
        if self._limiter is not None and reservation is not None:
            self._limiter.settle(reservation, completion.usage.total_tokens)
        return completion


def parse_completion(payload: Any) -> Completion:
    try:
        text = payload["choices"][0]["message"]["content"]
        usage = payload["usage"]
        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage["completion_tokens"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("unexpected response shape from the language model") from exc
    if not isinstance(text, str):
        raise ValueError("the language model returned a non-text answer")
    return Completion(text.strip(), Usage(prompt_tokens, completion_tokens))
