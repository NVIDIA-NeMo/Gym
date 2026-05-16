# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Pure-logic tests for the system-only splice helpers + the
``tokenize_system_block`` HTTP wrapper.

The ``tokenize_system_block`` test mocks ``ServerClient.post`` since the real
endpoint runs inside the model_server (vllm_model wrapper) at training time.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from responses_api_agents.aviary_agent_recontext.tokenizer_utils import (
    PrefixCheckResult,
    decode_snippet,
    responses_tools_to_chat_completion_tools,
    splice_system_block,
    tokenize_system_block,
    verify_system_prefix,
)


# ---------- verify_system_prefix ----------


class TestVerifySystemPrefix:
    def test_exact_match(self) -> None:
        result = verify_system_prefix([1, 2, 3, 9, 9], [1, 2, 3])
        assert result.ok
        assert result.divergence_index == -1
        assert result.reason == ""

    def test_full_sequence_match(self) -> None:
        result = verify_system_prefix([1, 2, 3], [1, 2, 3])
        assert result.ok

    def test_prompt_too_short(self) -> None:
        result = verify_system_prefix([1, 2], [1, 2, 3])
        assert not result.ok
        assert result.divergence_index == -1
        assert result.reason.startswith("prompt_too_short:")
        assert "2<3" in result.reason

    def test_divergence_first_token(self) -> None:
        result = verify_system_prefix([99, 2, 3, 9, 9], [1, 2, 3])
        assert not result.ok
        assert result.divergence_index == 0
        assert result.reason == "prefix_mismatch_at_index_0"

    def test_divergence_mid_block(self) -> None:
        result = verify_system_prefix([1, 2, 99, 9, 9], [1, 2, 3])
        assert not result.ok
        assert result.divergence_index == 2
        assert result.reason == "prefix_mismatch_at_index_2"


# ---------- splice_system_block ----------


class TestSpliceSystemBlock:
    def test_typical_splice(self) -> None:
        prompt = [1, 2, 3, 4, 5]
        new_sys = [9, 9, 9, 9]
        spliced = splice_system_block(prompt, orig_sys_len=3, new_sys_tokens=new_sys)
        assert spliced == [9, 9, 9, 9, 4, 5]

    def test_shrinking_system_block(self) -> None:
        prompt = [1, 2, 3, 4, 5]
        new_sys = [7]
        spliced = splice_system_block(prompt, orig_sys_len=3, new_sys_tokens=new_sys)
        assert spliced == [7, 4, 5]

    def test_full_consumed_no_tail(self) -> None:
        prompt = [1, 2, 3]
        spliced = splice_system_block(prompt, orig_sys_len=3, new_sys_tokens=[7, 7])
        assert spliced == [7, 7]

    def test_prompt_shorter_than_orig_sys_raises(self) -> None:
        with pytest.raises(ValueError, match="orig_sys_len"):
            splice_system_block([1, 2], orig_sys_len=3, new_sys_tokens=[9])


# ---------- decode_snippet ----------


class TestDecodeSnippet:
    def test_empty_returns_empty(self) -> None:
        class FakeTok:
            def decode(self, ids):
                return f"<{ids}>"
        assert decode_snippet(FakeTok(), [], around=0) == ""

    def test_no_tokenizer_returns_empty(self) -> None:
        assert decode_snippet(None, [1, 2, 3], around=1) == ""

    def test_decode_failure_returns_marker(self) -> None:
        class FakeTok:
            def decode(self, ids):
                raise RuntimeError("boom")
        out = decode_snippet(FakeTok(), [1, 2, 3], around=0, radius=2)
        assert out.startswith("<decode_failed:")


# ---------- responses_tools_to_chat_completion_tools ----------


class TestResponsesToolsToChatCompletionTools:
    """Mirrors vLLM wrapper's rollout-time conversion: Responses-API flat tools
    become Chat-Completions nested tools, with ``strict`` stripped."""

    def test_aviary_shape_converts(self) -> None:
        tools = [
            {
                "type": "function",
                "name": "calculator",
                "parameters": {"type": "object", "properties": {}},
                "strict": True,
                "description": "Calculate.",
            },
        ]
        out = responses_tools_to_chat_completion_tools(tools)
        assert out == [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "parameters": {"type": "object", "properties": {}},
                    "description": "Calculate.",
                },
            },
        ]

    def test_empty_passthrough(self) -> None:
        assert responses_tools_to_chat_completion_tools([]) == []

    def test_does_not_mutate_input(self) -> None:
        tools = [{"type": "function", "name": "f", "parameters": {}, "strict": True}]
        snapshot = [dict(t) for t in tools]
        responses_tools_to_chat_completion_tools(tools)
        assert tools == snapshot


# ---------- tokenize_system_block (HTTP wrapper) ----------


class TestTokenizeSystemBlock:
    """Ensures we POST to /v1/tokenize on the named model_server with a body
    matching the canonical chat-completions tokenize request shape."""

    @pytest.mark.asyncio
    async def test_basic_invocation(self) -> None:
        # Mock a ServerClient whose `post(...)` returns a response whose
        # `.json()` yields the tokenize endpoint shape.
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"tokens": [10, 11, 12], "count": 3})

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        tokens = await tokenize_system_block(
            mock_client,
            model_server_name="policy",
            system_text="You are helpful.",
        )

        assert tokens == [10, 11, 12]
        # Verify the request was constructed correctly.
        mock_client.post.assert_awaited_once()
        kwargs = mock_client.post.await_args.kwargs
        assert kwargs["server_name"] == "policy"
        assert kwargs["url_path"] == "/v1/tokenize"
        body = kwargs["json"]
        assert body["messages"] == [{"role": "system", "content": "You are helpful."}]
        assert body["add_generation_prompt"] is False
        assert "tools" not in body  # not passed → not in body

    @pytest.mark.asyncio
    async def test_tools_and_chat_template_kwargs_passed(self) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"tokens": [1]})

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        tools = [{"type": "function", "name": "f", "parameters": {}}]
        await tokenize_system_block(
            mock_client,
            "policy",
            "sys",
            tools=tools,
            chat_template_kwargs={"enable_thinking": True},
        )

        body = mock_client.post.await_args.kwargs["json"]
        assert body["tools"] == tools
        assert body["chat_template_kwargs"] == {"enable_thinking": True}

    @pytest.mark.asyncio
    async def test_missing_tokens_field_raises(self) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"count": 0})  # missing `tokens`

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="`tokens`"):
            await tokenize_system_block(mock_client, "policy", "sys")
