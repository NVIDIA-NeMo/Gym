# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest
from openai import APIStatusError, APITimeoutError, OpenAI

from resources_servers.aa_briefcase_lite import app
from resources_servers.gdpval.comparison import run_trials
from resources_servers.gdpval.judge_panel import ResolvedJudge


@pytest.mark.parametrize("status", [500, 503, None])
@pytest.mark.parametrize("recover", [False, True])
def test_pairwise_sdk_retries_are_bounded_and_preserve_input(monkeypatch, caplog, status, recover):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        if not recover or len(requests) < 3:
            if status is None:
                raise httpx.ReadTimeout("Connection timed out", request=request)
            return httpx.Response(status, json={"error": {"message": "Connection timed out", "code": "408"}})
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "BOXED[A]"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )

    client_class = app._PairwiseJudgeHttpClient
    monkeypatch.setattr(
        app, "_PairwiseJudgeHttpClient", lambda **kwargs: client_class(**kwargs, transport=httpx.MockTransport(handle))
    )
    monkeypatch.setattr(OpenAI, "_calculate_retry_timeout", lambda *args: 0)
    judges = app.AABriefcaseLiteResourcesServer._pairwise_judges(
        [ResolvedJudge(name="judge", model="test-model", base_url="http://test.invalid/v1", api_key="test-secret")]
    )
    kwargs = dict(
        judges=judges,
        task_prompt="private prompt",
        refs=[],
        submission_a=[],
        submission_b=[],
        num_trials=1,
        request_attempts=1,
        invalid_response_retries=2,
    )
    try:
        if recover:
            result = run_trials(**kwargs)
            assert result["win_count_a"] == 1
            assert result["invalid_count"] == 0
            assert "completion_tokens=20" in caplog.text
        else:
            with pytest.raises((APIStatusError, APITimeoutError)):
                run_trials(**kwargs)
        assert len(requests) == 3
        assert requests[0] == requests[1] == requests[2]
        assert "private prompt" not in caplog.text
        assert "test-secret" not in caplog.text
    finally:
        for judge in judges:
            judge.client.close()


@pytest.mark.parametrize("mode", ["binary", "pairwise"])
@pytest.mark.parametrize(
    "usage",
    [
        None,
        {
            "prompt_tokens": 100,
            "completion_tokens": 48,
            "total_tokens": 148,
            "completion_tokens_details": {"reasoning_tokens": 32},
            "prompt_tokens_details": {"cached_tokens": 80},
        },
    ],
)
async def test_usage_is_logged_before_answer_parsing_without_content(caplog, mode, usage):
    def handle(request):
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "private malformed answer"}, "finish_reason": "length"}],
                "usage": usage,
            },
        )

    if mode == "binary":
        async with app._BinaryJudgeHttpClient(
            check_id="check", model="test-model", transport=httpx.MockTransport(handle)
        ) as client:
            await client.post("http://test.invalid/v1/chat/completions", json={})
    else:
        with app._PairwiseJudgeHttpClient(model="test-model", transport=httpx.MockTransport(handle)) as client:
            client.post("http://test.invalid/v1/chat/completions", json={})
    assert f"mode={mode}" in caplog.text
    assert "finish_reason=length" in caplog.text
    assert f"completion_tokens={48 if usage else None}" in caplog.text
    assert f"reasoning_tokens={32 if usage else None}" in caplog.text
    assert "private malformed answer" not in caplog.text
