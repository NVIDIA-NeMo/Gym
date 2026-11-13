# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from harbor.llms.base import (
    ContextLengthExceededError,
    OutputLengthExceededError,
)

from responses_api_agents.terminus_2_agent.llm import NemoGymLLM


def _make_llm(**kwargs) -> NemoGymLLM:
    defaults = dict(model_name="test-model", api_base="http://localhost:8000/v1")
    defaults.update(kwargs)
    llm = NemoGymLLM(**defaults)
    llm._logger = logging.getLogger("test")
    return llm


def _mock_response(content="ok", finish_reason="stop", extra_message=None, extra_choice=None, **top_level):
    message = {"content": content}
    if extra_message:
        message.update(extra_message)
    choice = {"message": message, "finish_reason": finish_reason}
    if extra_choice:
        choice.update(extra_choice)
    resp = {"choices": [choice]}
    resp.update(top_level)
    return resp


async def _call(llm, mock_json, **call_kwargs):
    mock_post = AsyncMock(return_value=mock_json)
    with patch.object(llm, "_post_chat_completions", mock_post):
        response = await llm.call(**call_kwargs)
    return response, mock_post


@pytest.mark.asyncio
async def test_extracts_openai_shape():
    llm = _make_llm(collect_rollout_details=True)
    response, _ = await _call(
        llm,
        _mock_response(
            content="hello",
            id="chatcmpl-terminal",
            extra_message={"generation_token_ids": [7, 8]},
            extra_choice={"logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]}},
            prompt_token_ids=[1, 2, 3],
            usage={"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 4}},
        ),
        prompt="hello",
    )

    assert response.content == "hello"
    assert response.prompt_token_ids == [1, 2, 3]
    assert response.completion_token_ids == [7, 8]
    assert response.logprobs == [-0.1, -0.2]
    assert response.usage.prompt_tokens == 10
    assert response.usage.cache_tokens == 4
    assert llm.last_response_id == "chatcmpl-terminal"


@pytest.mark.asyncio
async def test_extracts_nemo_proxy_shape():
    llm = _make_llm(collect_rollout_details=True)
    routed_experts = [
        [[0, 1]],
        [[2, 3]],
    ]
    response, _ = await _call(
        llm,
        _mock_response(
            content="proxy output",
            extra_message={
                "prompt_token_ids": [11, 12],
                "generation_token_ids": ["token_id:13", "token_id:14"],
                "generation_log_probs": [-0.3, -0.4],
                "routed_experts": routed_experts,
            },
        ),
        prompt="hello",
    )

    assert response.prompt_token_ids == [11, 12]
    assert response.completion_token_ids == [13, 14]
    assert response.logprobs == [-0.3, -0.4]
    assert llm.pop_routed_experts_for_rollout_details([11, 12], [13, 14], [-0.3, -0.4]) == routed_experts
    assert llm.pop_routed_experts_for_rollout_details([11, 12], [13, 14], [-0.3, -0.4]) is None


@pytest.mark.asyncio
async def test_extracts_routed_experts_string_envelope():
    llm = _make_llm(collect_rollout_details=True)
    routed_experts = "nrlre1:int16:2x1x2:AAABAAIAAwA="
    response, _ = await _call(
        llm,
        _mock_response(
            content="proxy output",
            extra_message={
                "prompt_token_ids": [11, 12],
                "generation_token_ids": ["token_id:13", "token_id:14"],
                "generation_log_probs": [-0.3, -0.4],
                "routed_experts": routed_experts,
            },
        ),
        prompt="hello",
    )

    assert response.prompt_token_ids == [11, 12]
    assert llm.pop_routed_experts_for_rollout_details([11, 12], [13, 14], [-0.3, -0.4]) == routed_experts
    assert llm.pop_routed_experts_for_rollout_details([11, 12], [13, 14], [-0.3, -0.4]) is None


def test_duplicate_rollout_details_keys_do_not_guess_routed_experts():
    llm = _make_llm(collect_rollout_details=True)
    route_1 = [[[1]]]
    route_2 = [[[2]]]

    llm._store_routed_experts_for_rollout_details([1], [2], [-0.1], route_1)
    llm._store_routed_experts_for_rollout_details([1], [2], [-0.1], route_2)

    assert llm.pop_routed_experts_for_rollout_details([1], [2], [-0.1]) is None


@pytest.mark.asyncio
async def test_no_token_data_in_response():
    llm = _make_llm(collect_rollout_details=True)
    response, _ = await _call(llm, _mock_response(), prompt="hello")

    assert response.prompt_token_ids is None
    assert response.completion_token_ids is None
    assert response.logprobs is None


@pytest.mark.asyncio
async def test_collect_rollout_details_false_skips_extraction():
    llm = _make_llm(collect_rollout_details=False)
    response, _ = await _call(
        llm,
        _mock_response(
            extra_message={"generation_token_ids": [7, 8]},
            prompt_token_ids=[1, 2, 3],
        ),
        prompt="hello",
    )

    assert response.prompt_token_ids is None
    assert response.completion_token_ids is None


@pytest.mark.asyncio
async def test_on_policy_correction_attaches_token_ids():
    llm = _make_llm(collect_rollout_details=True)
    routed_experts = [
        [[0, 1]],
        [[2, 3]],
    ]

    await _call(
        llm,
        _mock_response(
            content="first",
            extra_message={"generation_token_ids": [10, 11], "routed_experts": routed_experts},
            prompt_token_ids=[1, 2, 3],
        ),
        prompt="hello",
    )

    _, mock_post = await _call(
        llm,
        _mock_response(content="second"),
        prompt="follow up",
        message_history=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "first"},
        ],
    )

    payload = mock_post.call_args[0][0]
    assistant_msg = [m for m in payload["messages"] if m["role"] == "assistant"][0]
    assert assistant_msg["prompt_token_ids"] == [1, 2, 3]
    assert assistant_msg["generation_token_ids"] == [10, 11]
    assert assistant_msg["routed_experts"] == routed_experts


@pytest.mark.asyncio
async def test_context_length_error_propagates():
    llm = _make_llm()
    with patch.object(llm, "_post_chat_completions", side_effect=ContextLengthExceededError("too long")):
        with pytest.raises(ContextLengthExceededError):
            await llm.call(prompt="hello")


@pytest.mark.asyncio
async def test_context_length_error_from_http_400():
    llm = _make_llm()
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(
        return_value=httpx.Response(
            status_code=400,
            text="maximum context length exceeded",
            request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"),
        )
    )

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(ContextLengthExceededError):
            await llm.call(prompt="hello")


@pytest.mark.asyncio
async def test_fake_response_id_raises_context_length_error():
    llm = _make_llm()
    with pytest.raises(ContextLengthExceededError, match="chtcmpl-123"):
        await _call(llm, _mock_response(content=None, id="chtcmpl-123"), prompt="hello")


@pytest.mark.asyncio
async def test_output_length_exceeded():
    llm = _make_llm()
    with pytest.raises(OutputLengthExceededError) as exc_info:
        await _call(llm, _mock_response(content="truncated", finish_reason="length"), prompt="hello")
    assert exc_info.value.truncated_response == "truncated"


@pytest.mark.asyncio
async def test_matched_think_tags():
    llm = _make_llm()
    response, _ = await _call(llm, _mock_response(content="<think>rc</think>text"), prompt="q")
    assert response.reasoning_content == "rc"
    assert response.content == "text"


@pytest.mark.asyncio
async def test_unmatched_close_tag():
    llm = _make_llm()
    response, _ = await _call(llm, _mock_response(content="rc</think>text"), prompt="q")
    assert response.reasoning_content == "rc"
    assert response.content == "text"


@pytest.mark.asyncio
async def test_server_reasoning_content_takes_precedence():
    llm = _make_llm()
    response, _ = await _call(
        llm,
        _mock_response(
            content="answer",
            extra_message={"reasoning_content": "server rc"},
        ),
        prompt="q",
    )
    assert response.reasoning_content == "server rc"
    assert response.content == "answer"


@pytest.mark.asyncio
async def test_entire_output_as_reasoning():
    llm = _make_llm()
    response, _ = await _call(llm, _mock_response(content="<think>all</think>"), prompt="q")
    assert response.content == "all"
    assert response.reasoning_content is None


@pytest.mark.asyncio
async def test_extra_chat_params_forwarded():
    llm = _make_llm(responses_create_params={"temperature": 0.5, "top_p": 0.9, "input": []})
    _, mock_post = await _call(llm, _mock_response(), prompt="hello")

    payload = mock_post.call_args[0][0]
    assert payload["temperature"] == 0.5
    assert payload["top_p"] == 0.9


@pytest.mark.asyncio
async def test_max_output_tokens_caps_each_model_call():
    llm = _make_llm(responses_create_params={"max_output_tokens": 32, "input": []})
    _, mock_post = await _call(llm, _mock_response(), prompt="hello")

    assert mock_post.call_args[0][0]["max_tokens"] == 32


def test_context_limit_from_max_input_tokens():
    assert _make_llm(model_info={"max_input_tokens": 32000}).get_model_context_limit() == 32000


def test_context_limit_falls_back_to_max_tokens():
    assert _make_llm(model_info={"max_tokens": 16000}).get_model_context_limit() == 16000


def test_context_limit_fallback_default():
    assert _make_llm(model_info={}).get_model_context_limit() == 1000000


def test_output_limit():
    assert _make_llm(model_info={"max_output_tokens": 8192}).get_model_output_limit() == 8192


def test_output_limit_none_when_missing():
    assert _make_llm(model_info={}).get_model_output_limit() is None


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        ("http://localhost:8000", "http://localhost:8000/v1/chat/completions"),
        ("http://localhost:8000/v1", "http://localhost:8000/v1/chat/completions"),
    ],
)
def test_chat_completions_endpoint(api_base, expected):
    assert _make_llm(api_base=api_base)._chat_completions_endpoint() == expected


@pytest.mark.parametrize("answered_first", [False, True])
async def test_pending_call_is_unknown_before_cancellation_and_after_it(answered_first):
    llm = _make_llm()
    if answered_first:
        response, _ = await _call(
            llm, _mock_response(usage={"prompt_tokens": 10, "completion_tokens": 2}), prompt="one"
        )
        assert response.usage.prompt_tokens == 10
    entered = asyncio.Event()

    async def pending(payload):
        entered.set()
        await asyncio.Event().wait()

    with patch.object(llm, "_post_chat_completions", side_effect=pending):
        task = asyncio.create_task(llm.call(prompt="pending"))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            # SIGTERM's snapshot occurs here, before the runner cancels the call.
            assert llm.calls_without_usage == 1
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert llm.calls_without_usage == 1


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": 10}, {"prompt_tokens": True, "completion_tokens": 2}])
async def test_partial_usage_does_not_clear_an_unaccounted_call(usage):
    llm = _make_llm()
    response, _ = await _call(llm, _mock_response(usage=usage), prompt="question")
    assert response.usage is None
    assert llm.calls_without_usage == 1


async def test_a_failed_attempt_remains_unknown_after_a_successful_retry():
    from tenacity import wait_none

    llm = _make_llm()
    response = _mock_response(usage={"prompt_tokens": 10, "completion_tokens": 2})
    with patch.object(llm, "_post_chat_completions", side_effect=[httpx.ReadTimeout("lost reply"), response]):
        result = await llm.call.retry_with(wait=wait_none())(llm, prompt="question")
    assert result.usage.prompt_tokens == 10
    assert llm.calls_without_usage == 1


def _http_response(status_code: int) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        text="model server error",
        request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"),
    )


@pytest.fixture
def retry_sleeps():
    """The backoff sleeps between attempts of ``NemoGymLLM.call``, recorded instead of slept."""
    sleeps: list[float] = []

    async def record(seconds: float) -> None:
        sleeps.append(seconds)

    with patch.object(NemoGymLLM.call.retry, "sleep", record):
        yield sleeps


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 413, 422])
async def test_a_non_transient_status_fails_on_the_first_attempt(status_code, retry_sleeps):
    llm = _make_llm()
    client = AsyncMock()
    client.post = AsyncMock(return_value=_http_response(status_code))

    with patch("httpx.AsyncClient", return_value=client):
        with pytest.raises(httpx.HTTPStatusError) as raised:
            await llm.call(prompt="hello")

    assert raised.value.response.status_code == status_code
    assert client.post.await_count == 1
    assert retry_sleeps == []


@pytest.mark.parametrize(
    "failure",
    [
        *(pytest.param(_http_response(code), id=str(code)) for code in (408, 425, 429, 500, 502, 503, 504)),
        pytest.param(httpx.ConnectError("connection refused"), id="refused"),
        pytest.param(httpx.ReadError("connection reset by peer"), id="reset"),
        pytest.param(httpx.RemoteProtocolError("server disconnected without sending a response"), id="dropped"),
        pytest.param(httpx.ReadTimeout("timed out"), id="timeout"),
    ],
)
async def test_a_transient_failure_is_retried_within_the_wait_bound(failure, retry_sleeps):
    llm = _make_llm()
    client = AsyncMock()
    if isinstance(failure, httpx.Response):
        client.post = AsyncMock(return_value=failure)
    else:
        client.post = AsyncMock(side_effect=failure)

    with patch("httpx.AsyncClient", return_value=client):
        with pytest.raises((httpx.HTTPStatusError, httpx.TransportError)):
            await llm.call(prompt="hello")

    assert client.post.await_count == 3
    assert len(retry_sleeps) == 2
    assert all(4 <= seconds <= 15 for seconds in retry_sleeps)
    assert sum(retry_sleeps) <= 30


async def test_a_transient_status_then_an_answer_returns_the_answer(retry_sleeps):
    llm = _make_llm()
    client = AsyncMock()
    answer = httpx.Response(
        status_code=200,
        json=_mock_response(content="recovered", id="chatcmpl-after-503"),
        request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"),
    )
    client.post = AsyncMock(side_effect=[_http_response(503), answer])

    with patch("httpx.AsyncClient", return_value=client):
        response = await llm.call(prompt="hello")

    assert response.content == "recovered"
    assert client.post.await_count == 2
    assert len(retry_sleeps) == 1


@pytest.mark.asyncio
async def test_missing_cache_detail_stays_unknown_after_a_later_reported_count():
    llm = _make_llm()
    for details in ({}, {"prompt_tokens_details": {"cached_tokens": 4}}):
        response, _ = await _call(
            llm,
            _mock_response(usage={"prompt_tokens": 10, "completion_tokens": 2, **details}),
            prompt="hello",
        )
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 2
        assert llm.cached_usage_known is False


@pytest.mark.asyncio
async def test_reported_zero_cache_detail_is_known():
    llm = _make_llm()
    response, _ = await _call(
        llm,
        _mock_response(
            usage={"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 0}}
        ),
        prompt="hello",
    )
    assert response.usage.cache_tokens == 0
    assert llm.cached_usage_known is True
