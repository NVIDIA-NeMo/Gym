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
from types import UnionType
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    NotRequired,
    Required,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import openai
import pytest
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming
from openai.types.responses import (
    EasyInputMessage,
    ResponseCodeInterpreterToolCall,
    ResponseComputerToolCall,
    ResponseCustomToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseReasoningItem,
)
from openai.types.responses.response_create_params import ResponseCreateParamsBase
from openai.types.responses.response_input_item import (
    AdditionalTools as InputAdditionalTools,
)
from openai.types.responses.response_input_item import (
    ComputerCallOutput as InputComputerCallOutput,
)
from openai.types.responses.response_input_item import (
    FunctionCallOutput as InputFunctionCallOutput,
)
from openai.types.responses.response_input_item import (
    Message as InputMessage,
)
from openai.types.responses.response_input_item import ResponseInputItem
from openai.types.responses.response_output_item import (
    ImageGenerationCall,
    LocalShellCall,
    McpApprovalRequest,
    McpCall,
    McpListTools,
)
from pydantic import ValidationError

from nemo_gym.openai_utils import (
    RESPONSES_TO_TRAIN,
    NeMoGymAsyncOpenAI,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionMessageCustomToolCall,
    NeMoGymChoice,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymImageGenerationCall,
    NeMoGymLocalShellCall,
    NeMoGymMessage,
    NeMoGymResponse,
    NeMoGymResponseAdditionalTools,
    NeMoGymResponseCodeInterpreterToolCall,
    NeMoGymResponseComputerCallOutput,
    NeMoGymResponseComputerToolCall,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseCustomToolCall,
    NeMoGymResponseFileSearchToolCall,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseFunctionWebSearch,
    NeMoGymResponseInputItem,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseMcpApprovalRequest,
    NeMoGymResponseMcpCall,
    NeMoGymResponseMcpListTools,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    TokenIDLogProbMixin,
    accumulate_response_usage,
    training_variant_of,
)
from nemo_gym.responses_converter import (
    _RESPONSE_NON_BOUNDARY_TYPES,
    _RESPONSE_OUTPUT_BOUNDARY_TYPES,
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

    @pytest.mark.parametrize(
        "role",
        ["unknown", "user", "assistant", "system", "critic", "discriminator", "developer", "tool"],
    )
    def test_additional_tools_output_preserved_then_normalized_for_replay(self, role: str) -> None:
        payload = {"type": "additional_tools", "id": "at_1", "role": role, "tools": []}

        response = NeMoGymResponse.model_validate(_response_with_output([payload]))
        assert isinstance(response.output[0], NeMoGymResponseAdditionalTools)
        assert response.output[0].role == role
        output_dump = response.model_dump(mode="json")["output"][0]
        assert output_dump["role"] == role

        replay = NeMoGymResponseCreateParamsNonStreaming(input=[output_dump])
        replay_dump = replay.model_dump(mode="json", exclude_unset=True)["input"][0]
        assert replay_dump["role"] == "developer"
        assert (
            InputAdditionalTools.model_validate(replay_dump).model_dump(mode="json", exclude_unset=True) == replay_dump
        )

    def test_failed_computer_output_preserved_then_status_removed_for_replay(self) -> None:
        payload = {
            "type": "computer_call_output",
            "id": "cco_1",
            "call_id": "call_1",
            "output": {"type": "computer_screenshot", "image_url": "data:image/png;base64,x"},
            "status": "failed",
        }

        response = NeMoGymResponse.model_validate(_response_with_output([payload]))
        assert isinstance(response.output[0], NeMoGymResponseComputerCallOutput)
        assert response.output[0].status == "failed"
        output_dump = response.model_dump(mode="json")["output"][0]
        assert output_dump["status"] == "failed"

        replay = NeMoGymResponseCreateParamsNonStreaming(input=[output_dump])
        replay_dump = replay.model_dump(mode="json", exclude_unset=True)["input"][0]
        assert "status" not in replay_dump
        assert (
            InputComputerCallOutput.model_validate(replay_dump).model_dump(mode="json", exclude_unset=True)
            == replay_dump
        )


class TestTokenMetadataValidation:
    @pytest.mark.parametrize(
        "token_metadata",
        [
            {"generation_token_ids": [2], "generation_log_probs": [-0.1]},
            {
                "prompt_token_ids": [1],
                "generation_token_ids": {"invalid": "shape"},
                "generation_log_probs": [-0.1],
            },
        ],
        ids=["partial", "malformed"],
    )
    def test_chat_request_rejects_invalid_metadata_instead_of_falling_back(self, token_metadata: dict) -> None:
        with pytest.raises(ValidationError):
            NeMoGymChatCompletionCreateParamsNonStreaming(
                messages=[
                    {
                        "role": "assistant",
                        "content": "answer",
                        **token_metadata,
                    }
                ]
            )

    def test_chat_response_rejects_partial_metadata_instead_of_falling_back(self) -> None:
        with pytest.raises(ValidationError):
            NeMoGymChoice(
                index=0,
                finish_reason="stop",
                message={
                    "role": "assistant",
                    "content": "answer",
                    "prompt_token_ids": [1],
                },
            )


class TestNeMoGymChatCompletionSchemas:
    def test_user_audio_and_file_content_parts_round_trip(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "UklGRg==", "format": "wav"},
                        },
                        {
                            "type": "file",
                            "file": {"file_id": "file-123"},
                        },
                    ],
                }
            ],
            "model": "gpt-test",
        }

        params = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(payload)
        round_tripped = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate_json(params.model_dump_json())

        assert round_tripped == params
        assert [part["type"] for part in params.messages[0]["content"]] == ["input_audio", "file"]

    def test_custom_tool_and_training_tool_call_round_trip(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "custom",
                            "custom": {"name": "shell", "input": "echo hello"},
                        }
                    ],
                    "prompt_token_ids": [1],
                    "generation_token_ids": [2],
                    "generation_log_probs": [-0.1],
                }
            ],
            "model": "gpt-test",
            "tools": [
                {
                    "type": "custom",
                    "custom": {
                        "name": "shell",
                        "description": "Run a shell command",
                        "format": {"type": "text"},
                    },
                }
            ],
        }

        params = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(payload)
        round_tripped = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate_json(params.model_dump_json())

        assert round_tripped == params
        assert params.tools[0]["type"] == "custom"
        assert params.messages[0]["tool_calls"][0]["type"] == "custom"
        assert params.messages[0]["generation_token_ids"] == [2]

    def test_custom_response_tool_call_round_trip(self) -> None:
        payload = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "custom",
                                "custom": {"name": "shell", "input": "echo hello"},
                            }
                        ],
                    },
                }
            ],
        }

        completion = NeMoGymChatCompletion.model_validate(payload)
        round_tripped = NeMoGymChatCompletion.model_validate_json(completion.model_dump_json())

        assert round_tripped == completion
        assert isinstance(completion.choices[0].message.tool_calls[0], NeMoGymChatCompletionMessageCustomToolCall)


class TestNeMoGymFunctionCallOutput:
    @pytest.mark.parametrize(
        "output",
        [
            "plain text",
            [{"type": "input_text", "text": "structured text"}],
            [{"type": "input_image", "image_url": "https://example.com/image.png", "detail": "high"}],
            [{"type": "input_file", "file_id": "file_123", "filename": "result.txt"}],
        ],
        ids=["string", "text", "image", "file"],
    )
    def test_accepts_and_preserves_openai_2_7_2_payloads(self, output) -> None:
        item = NeMoGymFunctionCallOutput(call_id="call_1", output=output)

        assert item.model_dump()["output"] == output


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

    def test_local_prompt_message_in_response_output_validates(self) -> None:
        response = NeMoGymResponse.model_validate(
            _response_with_output([{"type": "message", "role": "user", "content": "environment observation"}])
        )

        assert isinstance(response.output[0], NeMoGymEasyInputMessage)

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


def _usage(*, cached_tokens: int | None, reasoning_tokens: int | None) -> NeMoGymResponseUsage:
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


def test_accumulate_response_usage_keeps_unknown_details_unknown() -> None:
    known = _usage(cached_tokens=7, reasoning_tokens=4)
    unknown = _usage(cached_tokens=None, reasoning_tokens=None)

    forward = accumulate_response_usage(known, unknown)
    reverse = accumulate_response_usage(unknown, known)

    assert forward is not None
    assert reverse is not None
    assert forward.input_tokens_details.cached_tokens is None
    assert forward.output_tokens_details.reasoning_tokens is None
    assert reverse.input_tokens_details.cached_tokens is None
    assert reverse.output_tokens_details.reasoning_tokens is None


def test_accumulate_response_usage_tolerates_missing_detail_objects() -> None:
    first = _usage(cached_tokens=0, reasoning_tokens=1).model_copy(update={"input_tokens_details": None})
    second = _usage(cached_tokens=7, reasoning_tokens=4).model_copy(update={"output_tokens_details": None})

    result = accumulate_response_usage(first, second)

    assert result is not None
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (20, 10, 30)
    assert result.input_tokens_details is None
    assert result.output_tokens_details.reasoning_tokens == 1


# Responses item coverage against the installed OpenAI SDK.
#
# Gym defines its own item classes.
# It also maintains three lists that correspond to SDK union members:
# - ``NeMoGymResponseInputItem`` lists the item types Gym can represent.
# - ``_RESPONSE_OUTPUT_BOUNDARY_TYPES`` lists model-generated item types.
# - ``RESPONSES_TO_TRAIN`` maps item types that can carry sampled token IDs.
#
# An outdated list can cause non-streaming requests to fail.
# It can also cause streaming requests to omit transcript items.


# ``split_responses_input_output_items`` detects assistant messages by role.
# The boundary type set therefore excludes ``message``.
_BOUNDARY_EXEMPT = frozenset({"message"})


def _unwrap(annotation: Any) -> Any:
    """Strip ``Annotated`` wrappers.

    ``ResponseOutputItem`` is ``Annotated[Union[...], PropertyInfo]``.
    Gym's output item is ``Annotated[Union[...], BeforeValidator]``.
    Without unwrapping, ``get_args`` returns the ``Annotated`` arguments.
    It does not return the union members.
    """
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _union_members(annotation: Any) -> List[type]:
    annotation = _unwrap(annotation)
    args = get_args(annotation)
    return [_unwrap(arg) for arg in args] if args else [annotation]


def _literal_domain(annotation: Any) -> frozenset[Any] | None:
    """Return a finite top-level Literal domain after removing transparent wrappers.

    Return ``None`` when the annotation is not bounded by Literals.
    Do not descend into containers or nested models.
    """
    origin = get_origin(annotation)
    if origin in (Annotated, Required, NotRequired):
        return _literal_domain(get_args(annotation)[0])
    if origin is Literal:
        return frozenset(get_args(annotation))
    if origin in (Union, UnionType):
        domain: set[Any] = set()
        for member in get_args(annotation):
            if member is type(None):
                continue
            member_domain = _literal_domain(member)
            if member_domain is None:
                return None
            domain.update(member_domain)
        return frozenset(domain) if domain else None
    return None


def _field_annotations(model: type) -> Dict[str, Any]:
    """Read fields from either an SDK/Gym Pydantic model or SDK TypedDict."""
    model_fields = getattr(model, "model_fields", None)
    if model_fields is not None:
        return {name: field.annotation for name, field in model_fields.items()}
    return dict(getattr(model, "__annotations__", {}))


def _type_tags(model: type) -> List[str]:
    """The ``Literal`` values of a pydantic model's ``type`` field."""
    annotation = _field_annotations(model).get("type")
    if annotation is None:
        return []
    domain = _literal_domain(annotation)
    return [] if domain is None else [str(value) for value in domain]


def _tag_owners(union: Any) -> Dict[str, List[type]]:
    owners: Dict[str, List[type]] = {}
    for member in _union_members(union):
        for tag in _type_tags(member):
            owners.setdefault(tag, []).append(member)
    return owners


def _literal_fields(owners: List[type]) -> Dict[str, frozenset[Any] | None]:
    """Aggregate Literal domains accepted by all same-tag union alternatives.

    A ``None`` value means at least one owner accepts an unbounded annotation.
    Missing fields are omitted.
    """
    fields: Dict[str, frozenset[Any] | None] = {}
    for owner in owners:
        for field, annotation in _field_annotations(owner).items():
            domain = _literal_domain(annotation)
            if field not in fields:
                fields[field] = domain
            elif fields[field] is not None and domain is not None:
                fields[field] = fields[field] | domain
            else:
                fields[field] = None
    return fields


# Input and output ownership must remain independent.
# Some fields with the same type tag have different input and output ``Literal`` domains.
SDK_INPUT_TAG_OWNERS = _tag_owners(ResponseInputItem)
SDK_OUTPUT_TAG_OWNERS = _tag_owners(ResponseOutputItem)
GYM_INPUT_TAG_OWNERS = _tag_owners(NeMoGymResponseInputItem)
GYM_OUTPUT_TAG_OWNERS = _tag_owners(NeMoGymResponseOutputItem)
SDK_INPUT_TAGS = frozenset(SDK_INPUT_TAG_OWNERS)
SDK_OUTPUT_TAGS = frozenset(SDK_OUTPUT_TAG_OWNERS)

# Gym accepts provider-specific MCP status strings.
# Each exemption must identify a finite SDK ``Literal`` field.
_UNBOUNDED_GYM_INPUT_LITERAL_FIELDS = {
    ("mcp_call", "status"): "NVIDIA-hosted MCP endpoints may return statuses outside the SDK Literal.",
}

# Types Gym deliberately does not represent.
# ``item_reference`` points to an item held by the server.
# Gym keeps no item store.
# It cannot resolve the reference while replaying a transcript.
GYM_UNREPRESENTABLE_TYPES = frozenset({"item_reference"})


# Pair each manually defined Gym model with the SDK model it mirrors.
# ``_derived_hand_written_models`` identifies which Gym models require a pairing.
_HAND_WRITTEN_MODELS = [
    (NeMoGymEasyInputMessage, EasyInputMessage),
    (NeMoGymMessage, InputMessage),
    (NeMoGymResponseOutputMessage, ResponseOutputMessage),
    (NeMoGymResponseFunctionToolCall, ResponseFunctionToolCall),
    (NeMoGymFunctionCallOutput, InputFunctionCallOutput),
    (NeMoGymResponseReasoningItem, ResponseReasoningItem),
]

# Fields intentionally omitted from manually defined models.
# The OpenAI API can return ``None`` for ``NeMoGymResponseReasoningItem.status``.
# It rejects that field when Gym sends it in a later request.
_DELIBERATE_FIELD_OMISSIONS = {("NeMoGymResponseReasoningItem", "status")}


def _deliberately_omitted_tag_fields() -> set[tuple[str, str]]:
    models = {gym.__name__: gym for gym, _ in _HAND_WRITTEN_MODELS}
    return {
        (tag, field) for model_name, field in _DELIBERATE_FIELD_OMISSIONS for tag in _type_tags(models[model_name])
    }


def _derived_hand_written_models() -> List[type]:
    """Return union members that define SDK fields without subclassing an SDK model.

    A member with an OpenAI class in its MRO inherits SDK fields.
    A member that subclasses another Gym model inherits that model's fields.
    The remaining members define their own fields and require explicit parity checks.
    """
    union_members = set(_union_members(NeMoGymResponseInputItem)) | set(_union_members(NeMoGymResponseOutputItem))
    copies = [
        member
        for member in union_members
        if not any(base.__module__.startswith("openai.") for base in member.__mro__[1:])
    ]
    # A training variant inherits the fields of its base Gym model.
    # Checking the base model therefore covers the variant.
    return [model for model in copies if not any(other in model.__mro__[1:] for other in copies)]


def test_hand_written_model_list_is_complete() -> None:
    """Every hand-written union member must be paired with the SDK model it mirrors.

    The parity test iterates over ``_HAND_WRITTEN_MODELS``.
    It derives the Gym model set independently.
    Each SDK model pairing must be declared explicitly.
    """
    derived = {model.__name__ for model in _derived_hand_written_models()}
    declared = {gym.__name__ for gym, _ in _HAND_WRITTEN_MODELS}

    unpaired = sorted(derived - declared)
    assert not unpaired, (
        f"{unpaired} declare their own fields rather than subclassing an openai model, so they can "
        f"fall behind the SDK, and nothing checks them.\n"
        f"Fix: add each to _HAND_WRITTEN_MODELS with the SDK model it mirrors."
    )

    no_longer_hand_written = sorted(declared - derived)
    assert not no_longer_hand_written, (
        f"{no_longer_hand_written} are in _HAND_WRITTEN_MODELS but now subclass an openai model, "
        f"so they inherit its fields. Remove them from that list."
    )


@pytest.mark.parametrize("gym_model, sdk_model", _HAND_WRITTEN_MODELS, ids=lambda m: getattr(m, "__name__", m))
def test_hand_written_models_carry_every_sdk_field(gym_model: type, sdk_model: type) -> None:
    """A copied model must not fall behind the SDK model it copies.

    Union membership depends on the ``type`` discriminator.
    It does not verify field parity.
    Validation drops SDK fields that the corresponding Gym model omits.
    """
    missing = sorted(
        field
        for field in sdk_model.model_fields
        if field not in gym_model.model_fields and (gym_model.__name__, field) not in _DELIBERATE_FIELD_OMISSIONS
    )
    assert not missing, (
        f"{sdk_model.__name__} at openai {openai.__version__} has {missing} and "
        f"{gym_model.__name__} does not, so those fields are silently dropped.\n"
        f"Fix: add the field with the SDK's type, or record it in _DELIBERATE_FIELD_OMISSIONS "
        f"with the reason."
    )


def test_deliberate_omissions_are_still_real_sdk_fields() -> None:
    """Require each deliberate omission to identify an existing SDK field."""
    by_name = {gym.__name__: sdk for gym, sdk in _HAND_WRITTEN_MODELS}
    unknown_models = sorted(model for model, _ in _DELIBERATE_FIELD_OMISSIONS if model not in by_name)
    assert not unknown_models, f"Deliberate omissions name unknown Gym models: {unknown_models}"
    stale = sorted(
        f"{model}.{field}"
        for model, field in _DELIBERATE_FIELD_OMISSIONS
        if model in by_name and field not in by_name[model].model_fields
    )
    assert not stale, f"{stale} are recorded as deliberate omissions but the SDK no longer has them."


def test_sdk_union_is_introspectable() -> None:
    """Verify that the SDK union introspection returns members.

    The other parity tests would pass vacuously if this introspection returned no members.
    """
    for name, tags in (("ResponseInputItem", SDK_INPUT_TAGS), ("ResponseOutputItem", SDK_OUTPUT_TAGS)):
        assert len(tags) >= 13, (
            f"Only found {len(tags)} tagged members in {name} at openai {openai.__version__}. "
            f"The union shape probably changed and this file's introspection needs updating -- "
            f"do not relax this assertion."
        )
        assert "message" in tags
        assert "function_call" in tags


def test_gym_unions_are_introspectable_independently() -> None:
    """Verify input and output union introspection independently."""
    for name, owners in (
        ("NeMoGymResponseInputItem", GYM_INPUT_TAG_OWNERS),
        ("NeMoGymResponseOutputItem", GYM_OUTPUT_TAG_OWNERS),
    ):
        assert len(owners) >= 12, (
            f"Only found {len(owners)} tagged members in {name}. Its union shape probably changed "
            f"and this file's introspection needs updating -- do not point both sides at one alias."
        )
        assert all(tag and tagged_owners for tag, tagged_owners in owners.items())


def test_sdk_literal_domains_are_introspectable() -> None:
    """Verify that field introspection returns ``Literal`` domains."""
    for name, owners in (
        ("ResponseInputItem", SDK_INPUT_TAG_OWNERS),
        ("ResponseOutputItem", SDK_OUTPUT_TAG_OWNERS),
        ("NeMoGymResponseInputItem", GYM_INPUT_TAG_OWNERS),
        ("NeMoGymResponseOutputItem", GYM_OUTPUT_TAG_OWNERS),
    ):
        literal_fields = {
            (tag, field)
            for tag, tag_owners in owners.items()
            for field, domain in _literal_fields(tag_owners).items()
            if domain is not None
        }
        assert len(literal_fields) >= 20, (
            f"Only found {len(literal_fields)} top-level Literal fields in {name}; "
            f"the structural parity checks would be mostly vacuous."
        )
        assert ("message", "role") in literal_fields
        assert ("function_call", "type") in literal_fields


def test_literal_domain_unwraps_transparent_typing_layers() -> None:
    wrapped = Annotated[
        Union[Required[Literal["first"]], NotRequired[Literal["second"]], None],
        "metadata",
    ]
    assert _literal_domain(wrapped) == frozenset({"first", "second"})
    assert _literal_domain(Union[Literal["bounded"], str]) is None


@pytest.mark.parametrize("tag", sorted(SDK_OUTPUT_TAGS))
def test_gym_output_accepts_every_sdk_output_literal(tag: str) -> None:
    """Provider output Literal values must survive Gym response validation."""
    sdk_fields = _literal_fields(SDK_OUTPUT_TAG_OWNERS[tag])
    gym_fields = _literal_fields(GYM_OUTPUT_TAG_OWNERS.get(tag, []))
    for field, sdk_domain in sdk_fields.items():
        if sdk_domain is None:
            continue
        if (tag, field) in _deliberately_omitted_tag_fields():
            continue
        assert field in gym_fields, (
            f"openai {openai.__version__} output {tag}.{field} has Literal domain "
            f"{sorted(sdk_domain, key=repr)}, but no Gym output owner declares that field."
        )
        gym_domain = gym_fields[field]
        if gym_domain is None:  # An unbounded annotation accepts every provider value.
            continue
        missing = sdk_domain - gym_domain
        assert not missing, (
            f"openai {openai.__version__} can output {tag}.{field} values "
            f"{sorted(missing, key=repr)} that Gym rejects. SDK={sorted(sdk_domain, key=repr)}, "
            f"Gym output={sorted(gym_domain, key=repr)}."
        )


@pytest.mark.parametrize("tag", sorted(SDK_INPUT_TAGS))
def test_gym_input_literal_domains_match_sdk_input(tag: str) -> None:
    """Gym must accept the SDK input domain without admitting unsendable Literals."""
    if tag in GYM_UNREPRESENTABLE_TYPES:
        return
    sdk_fields = _literal_fields(SDK_INPUT_TAG_OWNERS[tag])
    gym_fields = _literal_fields(GYM_INPUT_TAG_OWNERS.get(tag, []))
    for field, sdk_domain in sdk_fields.items():
        if sdk_domain is None:
            continue
        if (tag, field) in _deliberately_omitted_tag_fields():
            continue
        assert field in gym_fields, (
            f"openai {openai.__version__} input {tag}.{field} has Literal domain "
            f"{sorted(sdk_domain, key=repr)}, but no Gym input owner declares that field."
        )
        gym_domain = gym_fields[field]
        if gym_domain is None:
            assert (tag, field) in _UNBOUNDED_GYM_INPUT_LITERAL_FIELDS, (
                f"Gym input {tag}.{field} is unbounded while the SDK domain is "
                f"{sorted(sdk_domain, key=repr)}. Narrow it to the SDK domain or explicitly "
                f"document the provider-tolerance widening."
            )
            continue
        missing = sdk_domain - gym_domain
        extra = gym_domain - sdk_domain
        assert not missing, f"Gym rejects SDK-supported input values for {tag}.{field}: {sorted(missing, key=repr)}."
        assert not extra, (
            f"Gym input {tag}.{field} accepts {sorted(extra, key=repr)}, but openai "
            f"{openai.__version__} input accepts only {sorted(sdk_domain, key=repr)}. "
            f"This usually means one Gym class is shared with a wider SDK output model. "
            f"Split the Gym input/output owners and normalize provider output before input validation."
        )
    gym_only_literals = {
        field: domain for field, domain in gym_fields.items() if domain is not None and field not in sdk_fields
    }
    assert not gym_only_literals, (
        f"Gym input {tag!r} declares Literal fields absent from the SDK input model: "
        f"{gym_only_literals}. These are usually output-only fields leaking through a shared owner."
    )


def test_shared_gym_owners_do_not_hide_unreplayable_output_literals() -> None:
    """A shared class cannot model output values that the SDK rejects as input."""
    for tag in sorted(SDK_OUTPUT_TAGS & SDK_INPUT_TAGS):
        output_fields = _literal_fields(SDK_OUTPUT_TAG_OWNERS[tag])
        input_fields = _literal_fields(SDK_INPUT_TAG_OWNERS[tag])
        shared_owners = set(GYM_OUTPUT_TAG_OWNERS.get(tag, [])) & set(GYM_INPUT_TAG_OWNERS.get(tag, []))
        for field, output_domain in output_fields.items():
            input_domain = input_fields.get(field, frozenset())
            if output_domain is None or input_domain is None:
                continue
            unreplayable = output_domain - input_domain
            if not unreplayable:
                continue
            unsafe_shared = [
                owner.__name__
                for owner in shared_owners
                if (domain := _literal_domain(_field_annotations(owner).get(field))) is None
                or bool(domain & unreplayable)
            ]
            assert not unsafe_shared, (
                f"openai {openai.__version__} output {tag}.{field} adds "
                f"{sorted(unreplayable, key=repr)}, which its input schema rejects, while Gym shares "
                f"{unsafe_shared} between input and output. Split the Gym owners, then normalize "
                f"provider output before validating replay input."
            )


def test_unbounded_input_literal_exemptions_are_live() -> None:
    """Require each exemption to match bounded SDK and unbounded Gym fields."""
    for tag, field in _UNBOUNDED_GYM_INPUT_LITERAL_FIELDS:
        assert tag in SDK_INPUT_TAG_OWNERS and tag in GYM_INPUT_TAG_OWNERS, (
            f"Unbounded input exemption {(tag, field)} names a tag absent from one input union."
        )
        sdk_domain = _literal_fields(SDK_INPUT_TAG_OWNERS[tag]).get(field)
        gym_domain = _literal_fields(GYM_INPUT_TAG_OWNERS[tag]).get(field, frozenset())
        assert sdk_domain is not None, (
            f"Unbounded input exemption {(tag, field)} is dead: SDK has no finite Literal domain."
        )
        assert gym_domain is None, f"Unbounded input exemption {(tag, field)} is dead: Gym is no longer unbounded."


def test_sdk_output_types_are_a_subset_of_input_types() -> None:
    """Verify that the input union includes every output item type.

    The tests use the input union as the source of item types.
    An output-only type would otherwise go unchecked.
    Gym could then fail when replaying that item as history.
    """
    emitted_but_not_sendable = sorted(set(SDK_OUTPUT_TAGS) - set(SDK_INPUT_TAGS))
    assert not emitted_but_not_sendable, (
        f"openai {openai.__version__} can return {emitted_but_not_sendable} but does not accept "
        f"them as input items, so driving these tests off ResponseInputItem no longer covers "
        f"everything. Check both unions here."
    )


def test_gym_output_union_represents_every_sdk_output_type() -> None:
    missing = sorted(set(SDK_OUTPUT_TAGS) - set(GYM_OUTPUT_TAG_OWNERS))
    unexpected = sorted(set(GYM_OUTPUT_TAG_OWNERS) - set(SDK_OUTPUT_TAGS))
    assert not (missing or unexpected), (
        f"NeMoGymResponseOutputItem differs from openai {openai.__version__} ResponseOutputItem: "
        f"missing={missing} unexpected={unexpected}"
    )


@pytest.mark.parametrize("tag", sorted(SDK_INPUT_TAGS))
def test_gym_union_represents_every_sdk_item_type(tag: str) -> None:
    """Require a Gym union member or explicit exclusion for each SDK input type.

    Without one, ``NeMoGymResponse.model_validate`` returns a 500 on the non-streaming path.
    ``sanitize_streaming_responses_body`` drops the item on the streaming path.
    """
    if tag in GYM_UNREPRESENTABLE_TYPES:
        assert tag not in GYM_INPUT_TAG_OWNERS, (
            f"{tag!r} is listed in GYM_UNREPRESENTABLE_TYPES but NeMoGymResponseInputItem now has "
            f"a member for it. Remove it from that list."
        )
        return
    assert tag in GYM_INPUT_TAG_OWNERS, (
        f"openai {openai.__version__} accepts a {tag!r} item and "
        f"NeMoGymResponseInputItem has no member for it.\n"
        f"Fix: add a NeMoGym* wrapper in nemo_gym/openai_utils.py and list it in "
        f"NeMoGymResponseInputItem. Then classify it in nemo_gym/responses_converter.py as "
        f"either _RESPONSE_OUTPUT_BOUNDARY_TYPES (the model generated it) or "
        f"_RESPONSE_NON_BOUNDARY_TYPES (a client-supplied result or bookkeeping).\n"
        f"If Gym should not represent it, add it to GYM_UNREPRESENTABLE_TYPES with the reason."
    )


@pytest.mark.parametrize("tag", sorted(SDK_INPUT_TAGS))
def test_every_sdk_item_type_is_classified(tag: str) -> None:
    """Classify each represented SDK item type by whether the model generated it.

    The classification cannot be read off the SDK.
    ``ResponseOutputItem`` can contain tool results and generated items.
    An unclassified type falls to the "not a boundary" side by default.
    That labels sampled tokens as prompt.
    """
    if tag in _BOUNDARY_EXEMPT or tag in GYM_UNREPRESENTABLE_TYPES:
        return
    is_boundary = tag in _RESPONSE_OUTPUT_BOUNDARY_TYPES
    is_not_boundary = tag in _RESPONSE_NON_BOUNDARY_TYPES
    assert is_boundary or is_not_boundary, (
        f"openai {openai.__version__} has an output item type {tag!r} that is in neither "
        f"_RESPONSE_OUTPUT_BOUNDARY_TYPES nor _RESPONSE_NON_BOUNDARY_TYPES "
        f"(nemo_gym/responses_converter.py).\n"
        f"Decide: did the model generate this item (boundary), or did the client supply it as a "
        f"tool result / approval / bookkeeping (not a boundary)? Defaulting is not safe."
    )
    assert not (is_boundary and is_not_boundary), (
        f"{tag!r} is in both the boundary and non-boundary sets; they must stay disjoint."
    )


def test_boundary_sets_are_disjoint() -> None:
    overlap = _RESPONSE_OUTPUT_BOUNDARY_TYPES & _RESPONSE_NON_BOUNDARY_TYPES
    assert not overlap, f"a type cannot be both a boundary and not a boundary: {sorted(overlap)}"


def test_message_is_not_in_the_boundary_set() -> None:
    """Keep user and system messages out of the generated output segment.

    ``split_responses_input_output_items`` handles assistant messages through its ``role == "assistant"`` check.
    Adding ``message`` to the type set would classify prompt messages as generated output.
    """
    assert "message" not in _RESPONSE_OUTPUT_BOUNDARY_TYPES


@pytest.mark.parametrize(
    "set_name, tags",
    [
        ("_RESPONSE_OUTPUT_BOUNDARY_TYPES", _RESPONSE_OUTPUT_BOUNDARY_TYPES),
        ("_RESPONSE_NON_BOUNDARY_TYPES", _RESPONSE_NON_BOUNDARY_TYPES),
        ("GYM_UNREPRESENTABLE_TYPES", GYM_UNREPRESENTABLE_TYPES),
        ("_BOUNDARY_EXEMPT", _BOUNDARY_EXEMPT),
    ],
)
def test_no_dead_entries_in_any_type_list(set_name: str, tags: frozenset) -> None:
    """Require every listed tag to be an item type in the installed SDK.

    An unknown tag never matches an item.
    The SDK-based parameterization also cannot exercise an unknown tag.
    """
    suspicious = tags - set(SDK_INPUT_TAGS)
    assert not suspicious, (
        f"{sorted(suspicious)} are in {set_name} but are not Responses item types at openai "
        f"{openai.__version__}. Either the tag is a typo, or it belongs to a later SDK and should "
        f"be added when the pin moves."
    )


def _classes_the_converter_can_emit() -> List[str]:
    """Return classes that can enter the converter's response output list.

    ``training_variant_of`` has one production caller, which passes it ``response_output[-1].__class__``.
    The AST identifies classes appended to that local list.
    """
    import ast
    import inspect

    from nemo_gym import responses_converter

    tree = ast.parse(inspect.getsource(responses_converter))
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "postprocess_assistant_message_dict"
    )

    # Map each local variable to the class used to construct it.
    local_classes = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name):
                        local_classes[target.id] = node.value.func.id

    emitted = set()
    for node in ast.walk(func):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        base = node.func.value
        if not (isinstance(base, ast.Name) and base.id == "response_output"):
            continue
        if node.func.attr not in ("append", "extend", "insert"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
                emitted.add(arg.func.id)
            elif isinstance(arg, ast.Name) and arg.id in local_classes:
                emitted.add(local_classes[arg.id])
            else:  # pragma: no cover - an unsupported AST expression
                emitted.add(f"<unresolved: {ast.dump(arg)[:60]}>")
    return sorted(emitted)


def test_every_item_the_converter_can_emit_has_a_training_variant() -> None:
    """Require a training variant for each class passed to ``training_variant_of``.

    Each variant is a member of ``NeMoGymResponseInputItem``.
    A type needs a variant when the converter can attach sampled token IDs to it.
    """
    emitted = _classes_the_converter_can_emit()
    unresolved = [name for name in emitted if name.startswith("<unresolved")]
    assert not unresolved, (
        f"this test's AST analysis could not resolve {unresolved} in "
        f"postprocess_assistant_message_dict, so it cannot prove the invariant. Extend "
        f"_classes_the_converter_can_emit rather than deleting the assertion."
    )

    registered = {cls.__name__ for cls in RESPONSES_TO_TRAIN}
    missing = sorted(set(emitted) - registered)
    assert not missing, (
        f"ResponsesConverter.postprocess_assistant_message_dict can emit {missing}, and "
        f"responses_converter.py hands response_output[-1] to training_variant_of(), so a "
        f"rollout carrying token IDs would fail there.\n"
        f"Fix: declare `class <Name>ForTraining(<Name>, TokenIDLogProbMixin)`, add it to "
        f"NeMoGymResponseInputItem, and register the pair in RESPONSES_TO_TRAIN."
    )


def test_training_variants_actually_carry_token_fields() -> None:
    """Require each registered variant to define the token payload fields."""
    for base, variant in RESPONSES_TO_TRAIN.items():
        assert issubclass(variant, base), f"{variant.__name__} must subclass {base.__name__}"
        assert issubclass(variant, TokenIDLogProbMixin), f"{variant.__name__} must mix in TokenIDLogProbMixin"
        for field in ("prompt_token_ids", "generation_token_ids", "generation_log_probs"):
            assert field in variant.model_fields, f"{variant.__name__} is missing {field}"


def test_training_variants_are_in_the_union() -> None:
    """Require each training variant to belong to the input union.

    A variant outside the union serializes but fails validation.
    """
    union_members = set(_union_members(NeMoGymResponseInputItem))
    missing = sorted(v.__name__ for v in RESPONSES_TO_TRAIN.values() if v not in union_members)
    assert not missing, f"ForTraining variants missing from NeMoGymResponseInputItem: {missing}"


def test_training_variant_lookup_fails_with_a_named_error() -> None:
    """Require a descriptive error for an unregistered class.

    A bare ``KeyError`` would produce an unexplained server error.
    """

    class _Unregistered:
        pass

    with pytest.raises(NotImplementedError, match="has no ForTraining variant"):
        training_variant_of(_Unregistered)


def test_duplicate_type_tags_are_documented() -> None:
    """Require explicit coverage for tags that map to several union members.

    Duplicate tags prevent use of ``Field(discriminator="type")``.
    They also make an unrecognized item report errors from every union member.
    """
    duplicates = {tag for tag, owners in GYM_INPUT_TAG_OWNERS.items() if len(owners) > 1}
    assert duplicates == {"message", "function_call", "reasoning"}, (
        f"duplicate type tags changed: {sorted(duplicates)}. Each duplicate widens the error "
        f"report for an unrecognised item; update this test if the change is intended."
    )


def test_request_model_carries_every_sdk_request_field() -> None:
    """Keep the strict request model aligned with the SDK request schema.

    Missing fields cause non-streaming validation errors.
    The streaming sanitizer would otherwise remove them.
    """
    sdk_fields = set(get_type_hints(ResponseCreateParamsBase, include_extras=True))
    missing = sorted(sdk_fields - set(NeMoGymResponseCreateParamsNonStreaming.model_fields))
    assert not missing, (
        f"openai {openai.__version__} accepts {missing} on a Responses request and "
        f"NeMoGymResponseCreateParamsNonStreaming does not, so those are rejected when sent "
        f"plainly and dropped when streaming.\n"
        f"Fix: mirror each field with the SDK's type."
    )


def test_chat_request_field_set_matches_sdk_without_deprecated_fields() -> None:
    """Keep the strict Chat request model aligned with the supported SDK fields.

    Deprecated fields remain disabled.
    """
    sdk_fields = set(get_type_hints(CompletionCreateParamsNonStreaming, include_extras=True))
    expected = sdk_fields - {"function_call", "functions"}
    actual = set(NeMoGymChatCompletionCreateParamsNonStreaming.model_fields)
    assert actual == expected, (
        f"openai {openai.__version__} Chat request fields changed: "
        f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
    )

    with pytest.raises(ValidationError):
        NeMoGymChatCompletionCreateParamsNonStreaming(messages=[], not_a_real_field=True)


def test_response_field_set_is_pinned() -> None:
    """Detect changes to the SDK ``Response`` fields.

    ``NeMoGymResponse`` inherits the SDK ``Response`` model.
    New SDK fields therefore appear in Gym model dumps.
    Update the expected set after deciding how Gym handles each changed field.
    """
    expected = {
        "background",
        "completed_at",
        "conversation",
        "created_at",
        "error",
        "id",
        "incomplete_details",
        "instructions",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "model",
        "moderation",
        "object",
        "output",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt",
        "prompt_cache_key",
        "prompt_cache_retention",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "status",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "truncation",
        "usage",
        "user",
    }
    actual = set(NeMoGymResponse.model_fields)
    added = sorted(actual - expected)
    removed = sorted(expected - actual)
    assert not (added or removed), (
        f"openai {openai.__version__} changed Response's field set: added={added} removed={removed}.\n"
        f"Decide what Gym does with each, then update this list."
    )
