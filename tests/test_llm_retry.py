"""Tests for shared LLM rate-limit retry helpers."""

from __future__ import annotations

import httpx
from openai import RateLimitError

from agent.llm_retry import invoke_with_rate_limit_retry


def _make_rate_limit_error(message: str = "Please try again in 1.25s.") -> RateLimitError:
    """Create a minimal OpenAI-style 429 error for retry testing."""

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request)
    return RateLimitError(message, response=response, body={})


def test_invoke_with_rate_limit_retry_retries_then_succeeds(monkeypatch) -> None:
    """The helper should sleep and retry when a temporary 429 occurs."""

    sleep_calls: list[float] = []
    monkeypatch.setattr("agent.llm_retry.time.sleep", sleep_calls.append)

    attempts = {"count": 0}

    def flaky_call() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise _make_rate_limit_error()
        return "ok"

    result = invoke_with_rate_limit_retry(
        flaky_call,
        max_attempts=4,
        base_delay_seconds=0.1,
        max_delay_seconds=1.0,
    )

    assert result == "ok"
    assert attempts["count"] == 3
    assert sleep_calls == [1.25, 1.25]


def test_invoke_with_rate_limit_retry_raises_after_last_attempt(monkeypatch) -> None:
    """The helper should re-raise if the rate limit never clears."""

    monkeypatch.setattr("agent.llm_retry.time.sleep", lambda _: None)

    def always_limited() -> str:
        raise _make_rate_limit_error("Please try again in 0.5s.")

    try:
        invoke_with_rate_limit_retry(
            always_limited,
            max_attempts=2,
            base_delay_seconds=0.1,
            max_delay_seconds=1.0,
        )
    except RateLimitError:
        pass
    else:
        raise AssertionError("Expected RateLimitError to be re-raised.")
