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
from openai.types.responses.response_output_item import (
    McpApprovalRequest,
    McpCall,
    McpListTools,
)
from pydantic import ValidationError

from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseMcpApprovalRequest,
    NeMoGymResponseMcpCall,
    NeMoGymResponseMcpListTools,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
    TokenIDLogProbMixin,
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


class TestAccumulateResponseUsage:
    @staticmethod
    def _usage(input_tokens: int, output_tokens: int, cached: int, reasoning: int) -> NeMoGymResponseUsage:
        return NeMoGymResponseUsage(
            input_tokens=input_tokens,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached),
            output_tokens=output_tokens,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=reasoning),
            total_tokens=input_tokens + output_tokens,
        )

    def test_sums_every_counter_including_details(self) -> None:
        # Multi-turn agents previously reset the details to 0 while summing the top-level counts,
        # which threw away the per-turn cache and reasoning counts the model server reported.
        total = self._usage(10, 5, cached=4, reasoning=2)
        accumulate_response_usage(total, self._usage(20, 7, cached=8, reasoning=3))

        assert total.input_tokens == 30
        assert total.output_tokens == 12
        assert total.total_tokens == 42
        assert total.input_tokens_details.cached_tokens == 12
        assert total.output_tokens_details.reasoning_tokens == 5

    def test_turn_without_detail_contributes_nothing(self) -> None:
        total = self._usage(10, 5, cached=4, reasoning=2)
        accumulate_response_usage(total, self._usage(1, 1, cached=0, reasoning=0))

        assert total.input_tokens_details.cached_tokens == 4
        assert total.output_tokens_details.reasoning_tokens == 2

    def test_does_not_mutate_the_addend(self) -> None:
        total = self._usage(10, 5, cached=4, reasoning=2)
        addend = self._usage(20, 7, cached=8, reasoning=3)
        accumulate_response_usage(total, addend)

        assert addend.input_tokens == 20
        assert addend.input_tokens_details.cached_tokens == 8
