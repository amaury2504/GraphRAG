"""Mistral chat completions (https://docs.mistral.ai/api/endpoint/chat)."""

import time
from collections.abc import Callable
from typing import Any

import httpx

from graphrag.adapters.http import send_with_retry
from graphrag.domain.models import Completion, Usage

API_URL = "https://api.mistral.ai/v1/chat/completions"
DEFAULT_MODEL = "mistral-small-latest"


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
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_attempts = max_attempts
        self._retry_delay = retry_delay
        self._max_delay = max_delay
        self._sleep = sleep

    def complete(self, system: str, user: str) -> Completion:
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
        return parse_completion(response.json())


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
