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
from typing import Annotated, Any, Dict, List, Literal, get_args, get_origin

import openai
import pytest
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
    NeMoGymResponseCodeInterpreterToolCall,
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


# ===========================================================================
# Responses item coverage against the installed openai SDK
#
# Gym declares its own NeMoGym* item classes rather than using the SDK's.
# It also keeps three hand-maintained lists derived from the same SDK union:
#   NeMoGymResponseInputItem, which types Gym can represent
#   _RESPONSE_OUTPUT_BOUNDARY_TYPES, which types start the model's generation
#   RESPONSES_TO_TRAIN, which types can carry sampled token IDs
#
# ResponseOutputItem gains members with most openai releases.
# None of the three lists is derived from it, so nothing else fails when they fall behind.
# Drift shows up as a 500 on the non-streaming path.
# On the streaming path it shows up as a silently truncated transcript.
#
# Run these against every version in the supported openai window, not only the pinned one.
# ===========================================================================


# `message` is covered by split_responses_input_output_items' `role == "assistant"` check.
# The type set does not carry it, so it is exempt from the boundary classification.
_BOUNDARY_EXEMPT = frozenset({"message"})


def _unwrap(annotation: Any) -> Any:
    """Strip ``Annotated`` wrappers.

    ``ResponseOutputItem`` is ``Annotated[Union[...], PropertyInfo]``.
    Gym's output item is ``Annotated[Union[...], BeforeValidator]``.
    Without unwrapping, ``get_args`` returns the two ``Annotated`` arguments instead of the union members.
    """
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _union_members(annotation: Any) -> List[type]:
    annotation = _unwrap(annotation)
    args = get_args(annotation)
    return [_unwrap(arg) for arg in args] if args else [annotation]


def _type_tags(model: type) -> List[str]:
    """The ``Literal`` values of a pydantic model's ``type`` field."""
    field = getattr(model, "model_fields", {}).get("type")
    if field is None:
        return []
    annotation = field.annotation
    if get_origin(annotation) is Literal:
        return [str(value) for value in get_args(annotation)]
    return [str(value) for arg in get_args(annotation) if get_origin(arg) is Literal for value in get_args(arg)]


def _sdk_tags(union: Any) -> Dict[str, type]:
    tags: Dict[str, type] = {}
    for member in _union_members(union):
        for tag in _type_tags(member):
            tags.setdefault(tag, member)
    return tags


def _gym_tag_owners() -> Dict[str, List[type]]:
    owners: Dict[str, List[type]] = {}
    for member in _union_members(NeMoGymResponseInputItem):
        for tag in _type_tags(member):
            owners.setdefault(tag, []).append(member)
    return owners


# ResponseInputItem is the source of truth rather than ResponseOutputItem.
# Gym has to carry every type a client can send, which is the wider set:
# test_sdk_output_types_are_a_subset_of_input_types keeps that relationship honest.
# An item type Gym cannot represent is a 500 on the non-streaming path either way.
SDK_INPUT_TAGS = _sdk_tags(ResponseInputItem)
SDK_OUTPUT_TAGS = _sdk_tags(ResponseOutputItem)
GYM_TAG_OWNERS = _gym_tag_owners()

# Types Gym deliberately does not represent.
# item_reference is a pointer to an item held server-side.
# Gym replays transcripts in full and keeps no item store, so it cannot resolve one.
# Rejecting the request is better than accepting a reference that resolves to nothing.
GYM_UNREPRESENTABLE_TYPES = frozenset({"item_reference"})


# The SDK model each hand-written Gym model mirrors.
# Only the pairing is declared here: which models are hand-written is derived from the union by
# _derived_hand_written_models(), and test_hand_written_model_list_is_complete fails if this list
# and that derivation disagree. Nothing here has to be remembered when a model is added.
_HAND_WRITTEN_MODELS = [
    (NeMoGymEasyInputMessage, EasyInputMessage),
    (NeMoGymMessage, InputMessage),
    (NeMoGymResponseOutputMessage, ResponseOutputMessage),
    (NeMoGymResponseFunctionToolCall, ResponseFunctionToolCall),
    (NeMoGymFunctionCallOutput, InputFunctionCallOutput),
    (NeMoGymResponseReasoningItem, ResponseReasoningItem),
]

# Fields left out of a hand-written model on purpose, with the reason.
# `NeMoGymResponseReasoningItem.status` is commented out at its definition: the OpenAI API returns
# None for it and then rejects the field when it is sent back on a later call in the same rollout.
_DELIBERATE_FIELD_OMISSIONS = {("NeMoGymResponseReasoningItem", "status")}


def _derived_hand_written_models() -> List[type]:
    """Union members that copy an SDK model instead of subclassing one.

    A member with an openai class in its MRO inherits new SDK fields, so it cannot fall behind.
    A member that subclasses another Gym copy is covered by whatever that copy declares.
    What is left declares its own fields against no SDK model, which is what can drift.
    """
    copies = [
        member
        for member in _union_members(NeMoGymResponseInputItem)
        if not any(base.__module__.startswith("openai.") for base in member.__mro__[1:])
    ]
    # A ForTraining variant subclasses the Gym model it adds token fields to, so it inherits
    # whatever that model declares and is covered by checking the model it derives from.
    return [model for model in copies if not any(other in model.__mro__[1:] for other in copies)]


def test_hand_written_model_list_is_complete() -> None:
    """Every hand-written union member must be paired with the SDK model it mirrors.

    _HAND_WRITTEN_MODELS is what the parity test iterates, so a hand-written model missing from it
    is simply never checked. The membership half is derived rather than remembered; only the
    pairing to an SDK model has to be written down, because nothing in the code records it.
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

    The union tests above match on the `type` discriminator, so a Gym model keeps owning its tag
    however far its fields drift. A field the SDK adds and Gym omits is dropped during validation
    without an error: the request is accepted, and the value is gone from everything downstream.
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
    """An omission recorded for a field the SDK no longer has hides the next real one."""
    by_name = {gym.__name__: sdk for gym, sdk in _HAND_WRITTEN_MODELS}
    stale = sorted(
        f"{model}.{field}"
        for model, field in _DELIBERATE_FIELD_OMISSIONS
        if model in by_name and field not in by_name[model].model_fields
    )
    assert not stale, f"{stale} are recorded as deliberate omissions but the SDK no longer has them."


def test_sdk_union_is_introspectable() -> None:
    """Guard the introspection itself.

    If a future SDK restructures either union so the helpers above find no members, every other
    test in this file would pass vacuously.
    """
    for name, tags in (("ResponseInputItem", SDK_INPUT_TAGS), ("ResponseOutputItem", SDK_OUTPUT_TAGS)):
        assert len(tags) >= 13, (
            f"Only found {len(tags)} tagged members in {name} at openai {openai.__version__}. "
            f"The union shape probably changed and this file's introspection needs updating -- "
            f"do not relax this assertion."
        )
        assert "message" in tags
        assert "function_call" in tags


def test_sdk_output_types_are_a_subset_of_input_types() -> None:
    """The input union is used as the source of truth, which relies on it being the wider set.

    A type the provider can return but a client cannot send would go unchecked.
    Gym would then 500 on replaying that item back as history, which is how issue #2436 happened.
    """
    emitted_but_not_sendable = sorted(set(SDK_OUTPUT_TAGS) - set(SDK_INPUT_TAGS))
    assert not emitted_but_not_sendable, (
        f"openai {openai.__version__} can return {emitted_but_not_sendable} but does not accept "
        f"them as input items, so driving these tests off ResponseInputItem no longer covers "
        f"everything. Check both unions here."
    )


@pytest.mark.parametrize("tag", sorted(SDK_INPUT_TAGS))
def test_gym_union_represents_every_sdk_item_type(tag: str) -> None:
    """Every type a client can send needs a Gym union member, or an explicit decision not to.

    Without one, ``NeMoGymResponse.model_validate`` returns a 500 on the non-streaming path.
    On the streaming path, ``sanitize_streaming_responses_body`` drops the item from the replayed
    transcript with a warning and returns 200.
    """
    if tag in GYM_UNREPRESENTABLE_TYPES:
        assert tag not in GYM_TAG_OWNERS, (
            f"{tag!r} is listed in GYM_UNREPRESENTABLE_TYPES but NeMoGymResponseInputItem now has "
            f"a member for it. Remove it from that list."
        )
        return
    assert tag in GYM_TAG_OWNERS, (
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
    """Each type Gym can carry is either a generation boundary or explicitly not one.

    The classification cannot be read off the SDK.
    Later versions put tool results in ``ResponseOutputItem`` alongside generated items.
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
    """A user or system message must not open the trained segment.

    ``split_responses_input_output_items`` handles assistant messages through its ``role == "assistant"`` check.
    Adding "message" to the type set would make the first prompt message a boundary.
    That classifies the whole prompt as generation.
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
    """Every tag named in any of these lists must be a Responses item type at the installed SDK.

    A tag that is not one never matches anything.
    It also makes the list look more complete than it is.
    Naming a type the pinned SDK lacks has the same effect, since nothing exercises the entry, so a
    type is added in the change that raises the pin far enough to introduce it.

    Every list is checked, because each one goes stale silently and in its own way:
    a dead entry in the classification sets leaves its intended type unclassified, and a dead entry
    in an exemption list leaves its intended type checked when it was meant to be skipped, or the
    reverse. In both cases the tests that would notice are parametrized over the SDK's tags, so they
    never visit a tag the SDK does not have.
    """
    suspicious = tags - set(SDK_INPUT_TAGS)
    assert not suspicious, (
        f"{sorted(suspicious)} are in {set_name} but are not Responses item types at openai "
        f"{openai.__version__}. Either the tag is a typo, or it belongs to a later SDK and should "
        f"be added when the pin moves."
    )


def _classes_the_converter_can_emit() -> List[str]:
    """Every class that can end up in postprocess_assistant_message_dict's output list.

    ``training_variant_of`` has one production caller, which passes it ``response_output[-1].__class__``.
    That list is local to the function.
    The classes that can reach the lookup are exactly those appended to it, which the AST can enumerate.
    This is derived rather than hard-coded, so adding a fourth ``append`` fails this test.
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

    # name -> class it is constructed from, for locals appended by reference
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
            else:  # pragma: no cover - a shape this analysis does not model
                emitted.add(f"<unresolved: {ast.dump(arg)[:60]}>")
    return sorted(emitted)


def test_every_item_the_converter_can_emit_has_a_training_variant() -> None:
    """Whatever can reach ``training_variant_of`` must be registered in RESPONSES_TO_TRAIN.

    Note this is not "every model-emitted type has a variant".
    Each variant is another member of the ``NeMoGymResponseInputItem`` smart union.
    An unrecognised item reports every member's errors.
    So a type gets a variant once a converter can emit it carrying sampled token IDs.
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
    """A registered variant must really add the token payload, not just be a distinct class."""
    for base, variant in RESPONSES_TO_TRAIN.items():
        assert issubclass(variant, base), f"{variant.__name__} must subclass {base.__name__}"
        assert issubclass(variant, TokenIDLogProbMixin), f"{variant.__name__} must mix in TokenIDLogProbMixin"
        for field in ("prompt_token_ids", "generation_token_ids", "generation_log_probs"):
            assert field in variant.model_fields, f"{variant.__name__} is missing {field}"


def test_training_variants_are_in_the_union() -> None:
    """A variant absent from the union cannot round-trip: it serializes but fails to validate."""
    union_members = set(_union_members(NeMoGymResponseInputItem))
    missing = sorted(v.__name__ for v in RESPONSES_TO_TRAIN.values() if v not in union_members)
    assert not missing, f"ForTraining variants missing from NeMoGymResponseInputItem: {missing}"


def test_training_variant_lookup_fails_with_a_named_error() -> None:
    """An unregistered class must not surface as a bare KeyError (a 500 with no explanation)."""

    class _Unregistered:
        pass

    with pytest.raises(NotImplementedError, match="has no ForTraining variant"):
        training_variant_of(_Unregistered)


def test_duplicate_type_tags_are_documented() -> None:
    """Record which tags map to several union members.

    These are why ``Field(discriminator="type")`` cannot be used.
    They are also why one unrecognised item produces errors from every union member.
    Pinning the expected set makes a new collision a deliberate change.
    """
    duplicates = {tag for tag, owners in GYM_TAG_OWNERS.items() if len(owners) > 1}
    assert duplicates == {"message", "function_call", "reasoning"}, (
        f"duplicate type tags changed: {sorted(duplicates)}. Each duplicate widens the error "
        f"report for an unrecognised item; update this test if the change is intended."
    )
