# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comparison judge requests retry throttling with backoff and nothing else."""

from unittest.mock import MagicMock

import httpx
import pytest
from openai import APITimeoutError

from resources_servers.gdpval import comparison


def _reply(text: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=text))]
    return response


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(comparison.time, "sleep", recorded.append)
    return recorded


def _client(*outcomes) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.side_effect = list(outcomes)
    return client


def test_throttling_is_retried_with_capped_exponential_backoff(sleeps):
    client = _client(*(RuntimeError("Error code: 429 - rate limit") for _ in range(4)), _reply(" BOXED[A] "))

    assert comparison.send_judge_request(client, "judge", [{"role": "user", "content": "x"}]) == "BOXED[A]"

    assert client.chat.completions.create.call_count == comparison.REQUEST_MAX_ATTEMPTS
    assert sleeps == [5.0, 10.0, 20.0, 40.0]


def test_throttling_gives_up_after_the_attempt_budget(sleeps):
    client = _client(*(RuntimeError("503 Service Unavailable") for _ in range(comparison.REQUEST_MAX_ATTEMPTS)))

    with pytest.raises(RuntimeError, match="503"):
        comparison.send_judge_request(client, "judge", [])

    assert client.chat.completions.create.call_count == comparison.REQUEST_MAX_ATTEMPTS
    assert len(sleeps) == comparison.REQUEST_MAX_ATTEMPTS - 1


@pytest.mark.parametrize(
    "error",
    [
        APITimeoutError(request=httpx.Request("POST", "http://judge.invalid/v1/chat/completions")),
        RuntimeError("Error code: 413 - request entity too large"),
        RuntimeError("Error code: 400 - maximum context length exceeded"),
        RuntimeError("Error code: 400 - invalid request"),
    ],
    ids=["timeout", "payload-too-large", "context-overflow", "bad-request"],
)
def test_non_retryable_errors_fail_on_the_first_attempt(sleeps, error):
    client = _client(error)

    with pytest.raises(type(error)):
        comparison.send_judge_request(client, "judge", [])

    assert client.chat.completions.create.call_count == 1
    assert sleeps == []


def test_member_overrides_replace_or_drop_defaults(sleeps):
    client = _client(_reply("BOXED[TIE]"))

    comparison.send_judge_request(
        client, "judge", [], create_overrides={"temperature": None, "reasoning_effort": "high"}
    )

    kwargs = client.chat.completions.create.call_args.kwargs
    assert "temperature" not in kwargs
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["max_tokens"] == 65535
