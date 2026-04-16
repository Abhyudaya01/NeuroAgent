"""Shared retry helpers for rate-limited chat-model calls."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from typing import TypeVar

from openai import RateLimitError


T = TypeVar("T")

DEFAULT_MAX_ATTEMPTS = 6
DEFAULT_BASE_DELAY_SECONDS = 2.0
DEFAULT_MAX_DELAY_SECONDS = 20.0


def _extract_retry_after_seconds(message: str) -> float | None:
    """Extract the server-suggested retry delay from a rate-limit error message."""

    match = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)s", message, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def invoke_with_rate_limit_retry(
    func: Callable[[], T],
    *,
    max_attempts: int | None = None,
    base_delay_seconds: float | None = None,
    max_delay_seconds: float | None = None,
) -> T:
    """Retry a chat-model call when OpenAI returns a temporary 429 TPM/RPM limit.

    The current benchmark runs can burst into short-lived token-per-minute limits.
    Instead of failing the whole agent run, back off briefly and retry a handful
    of times. Non-rate-limit exceptions are raised immediately.
    """

    attempts = max_attempts or int(os.getenv("OPENAI_RATE_LIMIT_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS))
    base_delay = (
        base_delay_seconds
        if base_delay_seconds is not None
        else float(os.getenv("OPENAI_RATE_LIMIT_BASE_DELAY_SECONDS", DEFAULT_BASE_DELAY_SECONDS))
    )
    max_delay = (
        max_delay_seconds
        if max_delay_seconds is not None
        else float(os.getenv("OPENAI_RATE_LIMIT_MAX_DELAY_SECONDS", DEFAULT_MAX_DELAY_SECONDS))
    )

    for attempt in range(1, attempts + 1):
        try:
            return func()
        except RateLimitError as error:
            if attempt >= attempts:
                raise
            suggested_delay = _extract_retry_after_seconds(str(error))
            exponential_delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            sleep_seconds = max(suggested_delay or 0.0, exponential_delay)
            time.sleep(sleep_seconds)

    raise RuntimeError("Rate-limit retry loop exited unexpectedly.")


__all__ = [
    "invoke_with_rate_limit_retry",
]
