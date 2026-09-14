# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rubric judge requests must be bounded, and must not retry.

Without an explicit timeout the OpenAI SDK applies its own default (600 s) plus
two silent retries, so a request that legitimately runs long burns three times
that before failing — and it surfaces as a bare transient 500 with nothing to
say a timeout was involved. That is how a task ends up unjudged three runs in a
row while every knob that looks like a timeout is set correctly somewhere else.
"""

from unittest.mock import AsyncMock, patch

import pytest

from resources_servers.gdpval import scoring


def _stub_judge_and_response(text: str):
    judge = type(
        "J",
        (),
        {"name": "stub", "model": "stub", "base_url": "http://stub", "api_key": "k", "create_overrides": {}},
    )()
    message = type("M", (), {"content": text, "tool_calls": None})()
    choice = type("C", (), {"message": message, "finish_reason": "stop"})()
    return judge, type("R", (), {"choices": [choice]})()


def test_timeout_is_configurable_and_not_the_sdk_default():
    assert scoring.JUDGE_REQUEST_TIMEOUT_SECONDS > 600, (
        "must exceed the SDK default, or a long judge request still dies at 600 s"
    )


async def _run_visual(tmp_path, judge, response):
    template = tmp_path / "judge.j2"
    template.write_text("{{ task_prompt }} {{ rubric }}")
    with patch("openai.AsyncOpenAI") as client_cls:
        client_cls.return_value.chat.completions.create = AsyncMock(return_value=response)
        await scoring.score_with_rubric_visual([], [], "rubric", "prompt", str(template), [judge])
    return client_cls


async def _run_structured(judge, response):
    with patch("openai.AsyncOpenAI") as client_cls:
        client_cls.return_value.chat.completions.create = AsyncMock(return_value=response)
        await scoring.score_with_rubric_structured(
            "text", [], "rubric", "prompt", [judge], num_trials=1, formatting_retries=1
        )
    return client_cls


@pytest.mark.asyncio
async def test_rubric_client_receives_timeout_and_no_retries(tmp_path):
    template = tmp_path / "judge.j2"
    template.write_text("{{ task_prompt }} {{ rubric }} {{ deliverable_text }}")
    judge, response = _stub_judge_and_response('{"overall_score": 0.5}')

    with patch("openai.AsyncOpenAI") as client_cls:
        client_cls.return_value.chat.completions.create = AsyncMock(return_value=response)
        await scoring.score_with_rubric("text", [], "rubric", "prompt", str(template), [judge])

    kwargs = client_cls.call_args.kwargs
    assert kwargs["timeout"] == scoring.JUDGE_REQUEST_TIMEOUT_SECONDS
    assert kwargs["max_retries"] == 0, "SDK retries triple the wall-clock cost of a slow request"


def test_environment_override_is_honoured(monkeypatch):
    monkeypatch.setenv("GDPVAL_JUDGE_REQUEST_TIMEOUT_SECONDS", "42")
    import importlib

    reloaded = importlib.reload(scoring)
    try:
        assert reloaded.JUDGE_REQUEST_TIMEOUT_SECONDS == 42.0
    finally:
        monkeypatch.delenv("GDPVAL_JUDGE_REQUEST_TIMEOUT_SECONDS", raising=False)
        importlib.reload(scoring)


@pytest.mark.asyncio
async def test_visual_client_is_bounded_too(tmp_path):
    """The multimodal path is where long requests actually happen."""
    judge, response = _stub_judge_and_response('{"overall_score": 0.5}')
    client_cls = await _run_visual(tmp_path, judge, response)

    kwargs = client_cls.call_args.kwargs
    assert kwargs["timeout"] == scoring.JUDGE_REQUEST_TIMEOUT_SECONDS
    assert kwargs["max_retries"] == 0


@pytest.mark.asyncio
async def test_structured_client_is_bounded_too():
    """Structured scoring caches a client per upstream; that one needs it as well."""
    judge, response = _stub_judge_and_response("FINAL_SCORE[8] out of MAX_POSSIBLE_SCORE[10]")
    client_cls = await _run_structured(judge, response)

    kwargs = client_cls.call_args.kwargs
    assert kwargs["timeout"] == scoring.JUDGE_REQUEST_TIMEOUT_SECONDS
    assert kwargs["max_retries"] == 0
