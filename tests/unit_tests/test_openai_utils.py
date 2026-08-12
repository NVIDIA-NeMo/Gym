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
import pytest
from aiohttp import ClientResponseError
from openai.types.responses import (
    ResponseCodeInterpreterToolCall,
    ResponseComputerToolCall,
    ResponseCustomToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_output_item import (
    ImageGenerationCall,
    LocalShellCall,
    McpApprovalRequest,
    McpCall,
    McpListTools,
)
from pydantic import ValidationError

from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymImageGenerationCall,
    NeMoGymLocalShellCall,
    NeMoGymResponse,
    NeMoGymResponseCodeInterpreterToolCall,
    NeMoGymResponseComputerToolCall,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseCustomToolCall,
    NeMoGymResponseFileSearchToolCall,
    NeMoGymResponseFunctionWebSearch,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseMcpApprovalRequest,
    NeMoGymResponseMcpCall,
    NeMoGymResponseMcpListTools,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
    TokenIDLogProbMixin,
    _non_retryable_rate_limit_reason,
    accumulate_response_usage,
)


def _response_with_output(output: list) -> dict:
    return {
        "id": "resp_1",
        "created_at": 0.0,
        "model": "gpt-oss-120b",
        "object": "response",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": output,
    }


class TestOpenAIUtils:
    async def test_NeMoGymAsyncOpenAI(self) -> None:
        NeMoGymAsyncOpenAI(api_key="abc", base_url="https://api.openai.com/v1")

    @pytest.mark.parametrize(
        ("content", "reason"),
        [
            ('{"error":{"code":"budget_exceeded","message":"limit"}}', "budget_exceeded"),
            ('{"error":{"code":"insufficient_quota","message":"limit"}}', "insufficient_quota"),
            ('{"error":{"message":"Budget has been exceeded!"}}', "quota_exhausted"),
        ],
    )
    def test_permanent_quota_429_is_not_retryable(self, content: str, reason: str) -> None:
        assert _non_retryable_rate_limit_reason(429, content) == reason

    def test_transient_rate_limit_remains_retryable(self) -> None:
        assert _non_retryable_rate_limit_reason(429, '{"error":{"code":"rate_limit"}}') is None
        assert _non_retryable_rate_limit_reason(503, '{"error":{"code":"budget_exceeded"}}') is None

    async def test_permanent_quota_response_fails_after_one_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        content = b'{"error":{"code":"budget_exceeded","message":"limit"}}'

        class FakeContent:
            async def read(self) -> bytes:
                return content

        class FakeResponse:
            status = 429
            content = FakeContent()

            def raise_for_status(self) -> None:
                raise ClientResponseError(None, (), status=self.status, message="Too Many Requests")

        calls = 0

        async def fake_request(**kwargs) -> FakeResponse:
            del kwargs
            nonlocal calls
            calls += 1
            return FakeResponse()

        monkeypatch.setattr("nemo_gym.openai_utils.request", fake_request)
        client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://example.test/v1")

        with pytest.raises(ClientResponseError) as error:
            await client._request_with_retry(method="POST", url="https://example.test/v1/responses")

        assert calls == 1
        assert error.value.response_content == content


class TestNeMoGymResponseCreateParamsNonStreaming:
    def test_seed_rejected_at_top_level(self) -> None:
        """seed is not part of the OpenAI Responses schema; it must be passed via metadata.extra_body."""
        with pytest.raises(ValidationError):
            NeMoGymResponseCreateParamsNonStreaming(input="hello", seed=42)

    def test_seed_via_metadata_extra_body(self) -> None:
        """seed passed through metadata.extra_body round-trips through the strict schema."""
        params = NeMoGymResponseCreateParamsNonStreaming(input="hello", metadata={"extra_body": '{"seed": 42}'})
        assert params.metadata["extra_body"] == '{"seed": 42}'

    def test_unknown_field_still_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            NeMoGymResponseCreateParamsNonStreaming(input="hello", not_a_real_field=1)


class TestNeMoGymResponseHostedMcpItems:
    """Hosted-MCP output items (``mcp_call`` etc.) must validate rather than 500.

    Endpoints that run tools server-side (e.g. NVIDIA-hosted gpt-oss surfacing
    its built-in python tool as MCP) emit these in ``response.output``; before
    they were in the union, ``NeMoGymResponse.model_validate`` raised and the
    model server returned a 500 that aborted the whole rollout collection.
    """

    def test_mcp_call_in_response_output_validates(self) -> None:
        mcp_call = {
            "type": "mcp_call",
            "id": "mcp_1",
            "name": "python",
            "server_label": "exec",
            "arguments": '{"code": "print(42)"}',
            "output": "42\n",
            "status": "completed",
        }
        response = NeMoGymResponse.model_validate(
            _response_with_output(
                [
                    {"type": "reasoning", "id": "r1", "summary": []},
                    mcp_call,
                    {
                        "type": "message",
                        "id": "m1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "(Answer: 42)", "annotations": []}],
                    },
                ]
            )
        )
        call = response.output[1]
        assert isinstance(call, NeMoGymResponseMcpCall)
        assert call.type == "mcp_call"
        assert call.output == "42\n"

    def test_mcp_call_tolerates_missing_optional_fields(self) -> None:
        call = NeMoGymResponseMcpCall.model_validate({"type": "mcp_call", "name": "python", "arguments": "{}"})
        assert call.id is None and call.server_label is None and call.output is None

    def test_mcp_list_tools_and_approval_request_validate(self) -> None:
        listing = NeMoGymResponseMcpListTools.model_validate(
            {"type": "mcp_list_tools", "id": "l1", "server_label": "s", "tools": [{"name": "python"}]}
        )
        approval = NeMoGymResponseMcpApprovalRequest.model_validate(
            {"type": "mcp_approval_request", "id": "a1", "name": "python", "arguments": "{}", "server_label": "s"}
        )
        assert listing.tools == [{"name": "python"}]
        assert approval.name == "python"

    def test_hosted_mcp_items_inherit_upstream_types(self) -> None:
        # These must inherit the upstream openai typing (only relaxing the fields
        # NVIDIA-hosted endpoints omit/widen) rather than redefine it from scratch.
        assert issubclass(NeMoGymResponseMcpCall, McpCall)
        assert issubclass(NeMoGymResponseMcpListTools, McpListTools)
        assert issubclass(NeMoGymResponseMcpApprovalRequest, McpApprovalRequest)


class TestNeMoGymResponseToolCallItems:
    """Responses API output-call items (``web_search_call`` etc.) must validate rather than 500.

    The OpenAI Responses API emits these in ``response.output`` for provider-
    executed tools and client-executed actions. Before they were in the union,
    ``NeMoGymResponse.model_validate`` raised and the model server returned a 500
    for an upstream response that succeeded (issue #2436).
    """

    def test_web_search_call_in_response_output_validates(self) -> None:
        response = NeMoGymResponse.model_validate(
            _response_with_output(
                [
                    {
                        "type": "web_search_call",
                        "id": "ws_1",
                        "action": {"type": "search", "query": "official OpenAI homepage domain"},
                        "status": "completed",
                    },
                    {"type": "reasoning", "id": "r1", "summary": []},
                    {
                        "type": "message",
                        "id": "m1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "openai.com", "annotations": []}],
                    },
                ]
            )
        )
        call = response.output[0]
        assert isinstance(call, NeMoGymResponseFunctionWebSearch)
        assert call.type == "web_search_call"
        assert call.status == "completed"

    def test_remaining_output_call_items_validate(self) -> None:
        response = NeMoGymResponse.model_validate(
            _response_with_output(
                [
                    {
                        "type": "file_search_call",
                        "id": "fs_1",
                        "queries": ["quarterly revenue"],
                        "status": "completed",
                    },
                    {
                        "type": "computer_call",
                        "id": "cu_1",
                        "call_id": "call_1",
                        "action": {"type": "screenshot"},
                        "pending_safety_checks": [],
                        "status": "completed",
                    },
                    {
                        "type": "image_generation_call",
                        "id": "ig_1",
                        "result": None,
                        "status": "completed",
                    },
                    {
                        "type": "code_interpreter_call",
                        "id": "ci_1",
                        "code": "print(42)",
                        "container_id": "cntr_1",
                        "outputs": [{"type": "logs", "logs": "42\n"}],
                        "status": "completed",
                    },
                    {
                        "type": "local_shell_call",
                        "id": "ls_1",
                        "call_id": "call_2",
                        "action": {"type": "exec", "command": ["echo", "42"], "env": {}},
                        "status": "completed",
                    },
                    {
                        "type": "custom_tool_call",
                        "id": "ct_1",
                        "call_id": "call_3",
                        "name": "my_tool",
                        "input": "{}",
                    },
                ]
            )
        )
        assert isinstance(response.output[0], NeMoGymResponseFileSearchToolCall)
        assert isinstance(response.output[1], NeMoGymResponseComputerToolCall)
        assert isinstance(response.output[2], NeMoGymImageGenerationCall)
        assert isinstance(response.output[3], NeMoGymResponseCodeInterpreterToolCall)
        assert isinstance(response.output[4], NeMoGymLocalShellCall)
        assert isinstance(response.output[5], NeMoGymResponseCustomToolCall)

    def test_output_call_items_accepted_as_input(self) -> None:
        # The upstream SDK also allows output-call items in ResponseInputItemParam:
        # a rollout echoes response.output back as input on the next turn, so
        # request validation must accept them too.
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "action": {"type": "search", "query": "official OpenAI homepage domain"},
                    "status": "completed",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What did you find?"}],
                },
            ]
        )
        assert isinstance(params.input[0], NeMoGymResponseFunctionWebSearch)

    def test_output_call_items_require_type_discriminator(self) -> None:
        with pytest.raises(ValidationError):
            NeMoGymResponse.model_validate(
                _response_with_output(
                    [
                        {
                            "id": "ws_1",
                            "action": {"type": "search", "query": "official OpenAI homepage domain"},
                            "status": "completed",
                        }
                    ]
                )
            )

        pairs = (
            (NeMoGymResponseFileSearchToolCall, ResponseFileSearchToolCall),
            (NeMoGymResponseFunctionWebSearch, ResponseFunctionWebSearch),
            (NeMoGymResponseComputerToolCall, ResponseComputerToolCall),
            (NeMoGymImageGenerationCall, ImageGenerationCall),
            (NeMoGymResponseCodeInterpreterToolCall, ResponseCodeInterpreterToolCall),
            (NeMoGymLocalShellCall, LocalShellCall),
            (NeMoGymResponseCustomToolCall, ResponseCustomToolCall),
        )
        assert all(gym_cls.model_fields["type"].is_required() for gym_cls, _ in pairs)

    def test_output_call_items_inherit_upstream_types(self) -> None:
        # These must inherit the upstream openai typing rather than redefine it
        # from scratch, so schema drift is caught when the openai pin moves.
        assert issubclass(NeMoGymResponseFileSearchToolCall, ResponseFileSearchToolCall)
        assert issubclass(NeMoGymResponseFunctionWebSearch, ResponseFunctionWebSearch)
        assert issubclass(NeMoGymResponseComputerToolCall, ResponseComputerToolCall)
        assert issubclass(NeMoGymImageGenerationCall, ImageGenerationCall)
        assert issubclass(NeMoGymResponseCodeInterpreterToolCall, ResponseCodeInterpreterToolCall)
        assert issubclass(NeMoGymLocalShellCall, LocalShellCall)
        assert issubclass(NeMoGymResponseCustomToolCall, ResponseCustomToolCall)


class TestRoutedExpertsWireFormats:
    _BASE = {
        "prompt_token_ids": [1, 2],
        "generation_token_ids": [3],
        "generation_log_probs": [-0.1],
    }

    def test_accepts_nested_int_lists(self) -> None:
        mixin = TokenIDLogProbMixin.model_validate({**self._BASE, "routed_experts": [[[0, 1]], [[2, 3]]]})
        assert mixin.routed_experts == [[[0, 1]], [[2, 3]]]

    def test_accepts_opaque_string_envelope(self) -> None:
        # Training frameworks may ship routes as a single opaque string (e.g. NeMo-RL's
        # "nrlre1:<dtype>:<SxLxK>:<base64>") so multi-MB payloads validate in O(1).
        envelope = "nrlre1:int16:2x1x2:AAABAAIAAwA="
        mixin = TokenIDLogProbMixin.model_validate({**self._BASE, "routed_experts": envelope})
        assert mixin.routed_experts == envelope

    def test_rejects_non_list_non_string(self) -> None:
        with pytest.raises(ValidationError):
            TokenIDLogProbMixin.model_validate({**self._BASE, "routed_experts": 42})


def _usage(*, cached_tokens: int, reasoning_tokens: int) -> NeMoGymResponseUsage:
    return NeMoGymResponseUsage(
        input_tokens=10,
        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached_tokens),
        output_tokens=5,
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=reasoning_tokens),
        total_tokens=15,
    )


def test_accumulate_response_usage_preserves_all_counts_and_missing_values() -> None:
    first = _usage(cached_tokens=0, reasoning_tokens=1)
    second = _usage(cached_tokens=7, reasoning_tokens=4)

    assert accumulate_response_usage(None, first) == first
    result = accumulate_response_usage(first, second)
    assert result is not None
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (20, 10, 30)
    assert (result.input_tokens_details.cached_tokens, result.output_tokens_details.reasoning_tokens) == (7, 5)
    assert first.input_tokens_details.cached_tokens == 0
    assert accumulate_response_usage(result, None) == result


def test_accumulate_response_usage_tolerates_missing_detail_objects() -> None:
    first = _usage(cached_tokens=0, reasoning_tokens=1).model_copy(update={"input_tokens_details": None})
    second = _usage(cached_tokens=7, reasoning_tokens=4).model_copy(update={"output_tokens_details": None})

    result = accumulate_response_usage(first, second)

    assert result is not None
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (20, 10, 30)
    assert result.input_tokens_details is None
    assert result.output_tokens_details.reasoning_tokens == 1
