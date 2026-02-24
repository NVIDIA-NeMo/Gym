import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from harbor.llms.base import (
    ContextLengthExceededError,
    OutputLengthExceededError,
)
from responses_api_agents.harbor_agent.custom_agents.llms.nemo_gym_llm import NemoGymLLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm(**kwargs) -> NemoGymLLM:
    defaults = dict(model_name="test-model", api_base="http://localhost:8000/v1")
    defaults.update(kwargs)
    llm = NemoGymLLM(**defaults)
    llm._logger = logging.getLogger("test")
    return llm


def _mock_response(content="ok", finish_reason="stop", extra_message=None, extra_choice=None, **top_level):
    """Build a minimal chat-completions response dict."""
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
    """Patch _post_chat_completions, call llm.call(), return (response, mock)."""
    mock_post = AsyncMock(return_value=mock_json)
    with patch.object(llm, "_post_chat_completions", mock_post):
        response = await llm.call(**call_kwargs)
    return response, mock_post


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extracts_openai_shape():
    """prompt_token_ids top-level, generation_token_ids in message, logprobs in choice."""
    llm = _make_llm(collect_rollout_details=True)
    response, _ = await _call(llm, _mock_response(
        content="hello",
        extra_message={"generation_token_ids": [7, 8]},
        extra_choice={"logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]}},
        prompt_token_ids=[1, 2, 3],
        usage={"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 4}},
    ), prompt="hello")

    assert response.content == "hello"
    assert response.prompt_token_ids == [1, 2, 3]
    assert response.completion_token_ids == [7, 8]
    assert response.logprobs == [-0.1, -0.2]
    assert response.usage.prompt_tokens == 10
    assert response.usage.cache_tokens == 4


@pytest.mark.asyncio
async def test_extracts_nemo_proxy_shape():
    """Token IDs and logprobs embedded in the message dict, string token_id format."""
    llm = _make_llm(collect_rollout_details=True)
    response, _ = await _call(llm, _mock_response(
        content="proxy output",
        extra_message={
            "prompt_token_ids": [11, 12],
            "generation_token_ids": ["token_id:13", "token_id:14"],
            "generation_log_probs": [-0.3, -0.4],
        },
    ), prompt="hello")

    assert response.prompt_token_ids == [11, 12]
    assert response.completion_token_ids == [13, 14]
    assert response.logprobs == [-0.3, -0.4]


@pytest.mark.asyncio
async def test_no_token_data_in_response():
    """When response has no token IDs / logprobs, fields are None."""
    llm = _make_llm(collect_rollout_details=True)
    response, _ = await _call(llm, _mock_response(), prompt="hello")

    assert response.prompt_token_ids is None
    assert response.completion_token_ids is None
    assert response.logprobs is None


@pytest.mark.asyncio
async def test_collect_rollout_details_false_skips_extraction():
    """Token IDs are not extracted when collect_rollout_details=False."""
    llm = _make_llm(collect_rollout_details=False)
    response, _ = await _call(llm, _mock_response(
        extra_message={"generation_token_ids": [7, 8]},
        prompt_token_ids=[1, 2, 3],
    ), prompt="hello")

    assert response.prompt_token_ids is None
    assert response.completion_token_ids is None


@pytest.mark.asyncio
async def test_on_policy_correction_attaches_token_ids():
    """After a call with rollout details, next call attaches token IDs to the last assistant message."""
    llm = _make_llm(collect_rollout_details=True)

    # First call — stores token IDs.
    await _call(llm, _mock_response(
        content="first",
        extra_message={"generation_token_ids": [10, 11]},
        prompt_token_ids=[1, 2, 3],
    ), prompt="hello")

    # Second call — includes prior assistant in history.
    _, mock_post = await _call(llm, _mock_response(content="second"), prompt="follow up", message_history=[
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "first"},
    ])

    payload = mock_post.call_args[0][0]
    assistant_msg = [m for m in payload["messages"] if m["role"] == "assistant"][0]
    assert assistant_msg["prompt_token_ids"] == [1, 2, 3]
    assert assistant_msg["generation_token_ids"] == [10, 11]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_length_error_propagates():
    """ContextLengthExceededError is not retried."""
    llm = _make_llm()
    with patch.object(llm, "_post_chat_completions", side_effect=ContextLengthExceededError("too long")):
        with pytest.raises(ContextLengthExceededError):
            await llm.call(prompt="hello")


@pytest.mark.asyncio
async def test_context_length_error_from_http_400():
    """HTTP 400 with context-length phrase raises ContextLengthExceededError."""
    llm = _make_llm()
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=httpx.Response(
        status_code=400,
        text="maximum context length exceeded",
        request=httpx.Request("POST", "http://localhost:8000/v1/chat/completions"),
    ))

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(ContextLengthExceededError):
            await llm.call(prompt="hello")


@pytest.mark.asyncio
async def test_fake_response_id_raises_context_length_error():
    """Gym proxy returns fake 200 with id='chtcmpl-123' for context-length overflow."""
    llm = _make_llm()
    with pytest.raises(ContextLengthExceededError, match="chtcmpl-123"):
        await _call(llm, _mock_response(content=None, id="chtcmpl-123"), prompt="hello")


@pytest.mark.asyncio
async def test_output_length_exceeded():
    """finish_reason='length' raises OutputLengthExceededError."""
    llm = _make_llm()
    with pytest.raises(OutputLengthExceededError) as exc_info:
        await _call(llm, _mock_response(content="truncated", finish_reason="length"), prompt="hello")
    assert exc_info.value.truncated_response == "truncated"


# ---------------------------------------------------------------------------
# Reasoning / think-tag extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_matched_think_tags():
    """<think>rc</think>text -> reasoning_content='rc', content='text'."""
    llm = _make_llm()
    response, _ = await _call(llm, _mock_response(content="<think>rc</think>text"), prompt="q")
    assert response.reasoning_content == "rc"
    assert response.content == "text"


@pytest.mark.asyncio
async def test_unmatched_close_tag():
    """rc</think>text (open tag in prompt) -> reasoning_content='rc', content='text'."""
    llm = _make_llm()
    response, _ = await _call(llm, _mock_response(content="rc</think>text"), prompt="q")
    assert response.reasoning_content == "rc"
    assert response.content == "text"


@pytest.mark.asyncio
async def test_server_reasoning_content_takes_precedence():
    """Server-provided reasoning_content skips tag parsing."""
    llm = _make_llm()
    response, _ = await _call(llm, _mock_response(
        content="answer", extra_message={"reasoning_content": "server rc"},
    ), prompt="q")
    assert response.reasoning_content == "server rc"
    assert response.content == "answer"


@pytest.mark.asyncio
async def test_entire_output_as_reasoning():
    """<think>all</think> with no remaining text -> content='all', reasoning=None."""
    llm = _make_llm()
    response, _ = await _call(llm, _mock_response(content="<think>all</think>"), prompt="q")
    assert response.content == "all"
    assert response.reasoning_content is None


# ---------------------------------------------------------------------------
# Extra chat params
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extra_chat_params_forwarded():
    """responses_create_params are forwarded in the chat payload."""
    llm = _make_llm(responses_create_params={"temperature": 0.5, "top_p": 0.9, "input": []})
    _, mock_post = await _call(llm, _mock_response(), prompt="hello")

    payload = mock_post.call_args[0][0]
    assert payload["temperature"] == 0.5
    assert payload["top_p"] == 0.9


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("api_base", "expected"), [
    ("http://localhost:8000", "http://localhost:8000/v1/chat/completions"),
    ("http://localhost:8000/v1", "http://localhost:8000/v1/chat/completions"),
])
def test_chat_completions_endpoint(api_base, expected):
    assert _make_llm(api_base=api_base)._chat_completions_endpoint() == expected
