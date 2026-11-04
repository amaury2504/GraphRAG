import json
from typing import Any

import httpx
import pytest

from graphrag.adapters.http import RequestFailedError
from graphrag.adapters.mistral import API_URL, MistralLLM, mistral_client, parse_completion
from graphrag.domain.models import Completion, Usage

KEY = "sk-test-secret-key"
OK = {
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "  Hello [1].  "}}],
    "usage": {"prompt_tokens": 17, "completion_tokens": 2, "total_tokens": 19},
}


class Env:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    def llm(self, **kwargs: Any) -> MistralLLM:
        client = mistral_client(KEY, transport=httpx.MockTransport(self.handler))
        return MistralLLM(client, sleep=self.sleeps.append, **kwargs)


def test_request_shape_and_parsed_answer() -> None:
    env = Env([httpx.Response(200, json=OK)])
    completion = env.llm(model="m-test", max_tokens=123, temperature=0.2).complete("SYS", "USER")
    assert completion == Completion("Hello [1].", Usage(17, 2))
    (request,) = env.requests
    assert (request.method, str(request.url)) == ("POST", API_URL)
    assert json.loads(request.content) == {
        "model": "m-test",
        "temperature": 0.2,
        "max_tokens": 123,
        "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USER"},
        ],
    }


def test_defaults_are_small_model_deterministic_and_bounded() -> None:
    env = Env([httpx.Response(200, json=OK)])
    env.llm().complete("s", "u")
    body = json.loads(env.requests[0].content)
    assert (body["model"], body["temperature"], body["max_tokens"]) == (
        "mistral-small-latest",
        0.0,
        700,
    )


def test_key_travels_in_the_authorization_header_only() -> None:
    env = Env([httpx.Response(200, json=OK)])
    env.llm().complete("s", "u")
    (request,) = env.requests
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert KEY not in str(request.url)
    assert KEY.encode() not in request.content


def test_rate_limits_are_retried_with_backoff() -> None:
    env = Env([httpx.Response(429), httpx.Response(503), httpx.Response(200, json=OK)])
    assert env.llm().complete("s", "u").text == "Hello [1]."
    assert len(env.requests) == 3
    assert env.sleeps == [2.0, 4.0]


def test_gives_up_after_the_configured_attempts_without_leaking_the_key() -> None:
    env = Env([httpx.Response(429)] * 3)
    with pytest.raises(RequestFailedError, match="3 attempts") as caught:
        env.llm(max_attempts=3).complete("s", "u")
    assert KEY not in str(caught.value)


def test_client_errors_are_not_retried() -> None:
    env = Env([httpx.Response(401, json={"message": "Unauthorized"})])
    with pytest.raises(RequestFailedError, match="HTTP 401"):
        env.llm().complete("s", "u")
    assert len(env.requests) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": [], "usage": OK["usage"]},
        {"choices": [{"message": {}}], "usage": OK["usage"]},
        {"choices": OK["choices"]},
        {"choices": OK["choices"], "usage": {"prompt_tokens": 1}},
        {"choices": OK["choices"], "usage": {"prompt_tokens": "x", "completion_tokens": 1}},
        {"choices": [{"message": {"content": ["chunk"]}}], "usage": OK["usage"]},
        {"choices": [{"message": {"content": None}}], "usage": OK["usage"]},
        None,
    ],
)
def test_unexpected_payloads_are_rejected(payload: Any) -> None:
    with pytest.raises(ValueError, match="language model"):
        parse_completion(payload)


def _big_ok(total: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": total - 1, "completion_tokens": 1, "total_tokens": total},
        },
    )


def test_calls_are_spread_to_stay_under_the_token_quota() -> None:
    now = [0.0]
    env = Env([_big_ok(900), _big_ok(900)])

    def sleep(seconds: float) -> None:
        env.sleeps.append(seconds)
        now[0] += seconds

    client = mistral_client(KEY, transport=httpx.MockTransport(env.handler))
    llm = MistralLLM(
        client, max_tokens=200, tokens_per_minute=1000, sleep=sleep, clock=lambda: now[0]
    )
    llm.complete("s", "u")  # real usage 900 replaces the small estimate
    llm.complete("s", "u")  # 900 + estimate would exceed 1000: wait for the window
    assert env.sleeps == [60.0]


def test_the_limiter_can_be_turned_off() -> None:
    env = Env([_big_ok(900), _big_ok(900)])
    llm = env.llm(tokens_per_minute=None)
    llm.complete("s", "u")
    llm.complete("s", "u")
    assert env.sleeps == []
