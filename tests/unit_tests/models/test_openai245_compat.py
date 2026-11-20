# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""OpenAI 2.45 wire fields survive Gym validation, replay and boundary splitting."""

from copy import deepcopy

import openai
import pytest
from pydantic import ValidationError

from nemo_gym.core.responses_api_types import (
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseProgram,
    NeMoGymResponseProgramOutput,
    training_variant_of,
)
from nemo_gym.models.anthropic_converter import AnthropicConverter
from nemo_gym.models.chat_completions_converter import (
    ChatCompletionsConverter,
    ChatCompletionsConverterState,
    split_responses_input_output_items,
)


PROGRAM = {
    "type": "program",
    "id": "program_synthetic",
    "call_id": "program_call_synthetic",
    "code": "return 1;",
    "fingerprint": 'opaque-λ\n{"not":"parsed"};fingerprint',
}
PROGRAM_OUTPUT = {
    "type": "program_output",
    "id": "program_result_synthetic",
    "call_id": "program_call_synthetic",
    "result": "1",
    "status": "completed",
}
FUNCTION_CALL = {"type": "function_call", "name": "lookup", "arguments": "{}", "call_id": "lookup-call"}
FUNCTION_OUTPUT = {"type": "function_call_output", "call_id": "lookup-call", "output": "synthetic result"}
CUSTOM_CALL = {"type": "custom_tool_call", "name": "custom", "input": "text", "call_id": "custom-call"}
CUSTOM_OUTPUT = {
    "type": "custom_tool_call_output",
    "call_id": "custom-call",
    "output": "synthetic result",
    "id": "custom-result",
    "status": "completed",
}
CALLER_ITEMS = [FUNCTION_CALL, FUNCTION_OUTPUT, CUSTOM_CALL, CUSTOM_OUTPUT]


def test_openai_sdk_version_is_the_audited_version() -> None:
    assert openai.__version__ == "2.45.0"


def test_programmatic_tools_and_tool_choice_survive_replay() -> None:
    payload = {
        "input": "synthetic task",
        "tools": [
            {"type": "programmatic_tool_calling"},
            {
                "type": "function",
                "name": "lookup",
                "parameters": {},
                "strict": False,
                "allowed_callers": ["direct", "programmatic"],
            },
        ],
        "tool_choice": {"type": "programmatic_tool_calling"},
    }
    request = NeMoGymResponseCreateParamsNonStreaming.model_validate(payload)
    dumped = request.model_dump(mode="json", exclude_unset=True)
    assert dumped == payload
    assert (
        NeMoGymResponseCreateParamsNonStreaming.model_validate(dumped).model_dump(mode="json", exclude_unset=True)
        == payload
    )


@pytest.mark.parametrize("caller", ["program", "invalid"])
def test_programmatic_tool_callers_are_not_widened(caller: str) -> None:
    with pytest.raises(ValidationError):
        NeMoGymResponseCreateParamsNonStreaming(
            input="synthetic task",
            tools=[
                {"type": "function", "name": "lookup", "parameters": {}, "strict": False, "allowed_callers": [caller]}
            ],
        )


def _response(items: list[dict]) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "resp_synthetic",
            "created_at": 0,
            "model": "synthetic",
            "object": "response",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": items,
        }
    )


def test_program_items_preserve_opaque_fingerprint_through_output_and_replay() -> None:
    items = [PROGRAM, PROGRAM_OUTPUT]
    parsed = _response(items)
    assert isinstance(parsed.output[0], NeMoGymResponseProgram)
    assert isinstance(parsed.output[1], NeMoGymResponseProgramOutput)
    serialized = parsed.model_dump(mode="json", exclude_unset=True)["output"]
    assert serialized == items
    replay = NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": serialized})
    assert isinstance(replay.input[0], NeMoGymResponseProgram)
    assert isinstance(replay.input[1], NeMoGymResponseProgramOutput)
    assert replay.model_dump(mode="json", exclude_unset=True)["input"] == items
    assert replay.input[0].fingerprint == PROGRAM["fingerprint"]


def test_program_opens_generated_segment_without_reordering_program_results() -> None:
    prior = PROGRAM_OUTPUT | {"id": "prior_result", "call_id": "prior_call"}
    items = _response([prior, PROGRAM, PROGRAM_OUTPUT]).output
    prompt, generated = split_responses_input_output_items(items)
    assert [item.model_dump(mode="json", exclude_unset=True) for item in prompt] == [prior]
    assert [item.model_dump(mode="json", exclude_unset=True) for item in generated] == [
        PROGRAM,
        PROGRAM_OUTPUT,
    ]
    prompt_only, generated_only = split_responses_input_output_items(_response([prior]).output)
    assert [item.model_dump(mode="json", exclude_unset=True) for item in prompt_only] == [prior]
    assert generated_only == []


@pytest.mark.parametrize("item_class", [NeMoGymResponseProgram, NeMoGymResponseProgramOutput])
def test_provider_program_items_do_not_invent_training_variants(item_class: type) -> None:
    with pytest.raises(NotImplementedError, match="has no ForTraining variant"):
        training_variant_of(item_class)


@pytest.mark.parametrize(
    "item, missing",
    [(PROGRAM, "fingerprint"), (PROGRAM_OUTPUT, "result")],
)
def test_program_items_retain_required_sdk_fields(item: dict, missing: str) -> None:
    malformed = {key: value for key, value in item.items() if key != missing}
    with pytest.raises(ValidationError):
        _response([malformed])
    with pytest.raises(ValidationError):
        NeMoGymResponseCreateParamsNonStreaming(input=[malformed])


@pytest.mark.parametrize(
    "item", CALLER_ITEMS, ids=["function-call", "function-result", "custom-call", "custom-result"]
)
@pytest.mark.parametrize("caller", [{"type": "program", "caller_id": "synthetic-private"}])
@pytest.mark.parametrize("drop", [False, True])
def test_chat_conversion_rejects_caller_linkage_even_when_reporting_fields_may_drop(
    item: dict, caller: dict, drop: bool
) -> None:
    converter = ChatCompletionsConverter(return_token_id_information=False, custom_tool_strategy="function_bridge")
    request = NeMoGymResponseCreateParamsNonStreaming(input=[{**item, "caller": caller}])
    before = request.model_dump(mode="json")
    with pytest.raises(NotImplementedError, match="^Responses .*caller metadata") as failure:
        converter.responses_to_chat_completion_create_params(request, drop_unrepresentable_request_fields=drop)
    assert "synthetic-private" not in str(failure.value)
    assert request.model_dump(mode="json") == before


@pytest.mark.parametrize(
    "item", CALLER_ITEMS, ids=["function-call", "function-result", "custom-call", "custom-result"]
)
def test_caller_rejection_does_not_change_conversion_state(item: dict) -> None:
    converter = ChatCompletionsConverter(return_token_id_information=False)
    state = ChatCompletionsConverterState(
        return_token_id_information=False, assistant_item_buffered=True, content_buffer="buffered text"
    )
    before = deepcopy(state)
    method = {
        "function_call": converter._format_function_call,
        "custom_tool_call": converter._format_custom_tool_call,
        "function_call_output": converter._format_function_call_output,
        "custom_tool_call_output": converter._format_function_call_output,
    }[item["type"]]
    with pytest.raises(NotImplementedError, match="^Responses .*caller metadata"):
        method({**item, "caller": {"type": "program", "caller_id": "p"}}, state)
    assert state == before


@pytest.mark.parametrize("caller", [None, {"type": "direct"}])
def test_absent_null_and_direct_callers_keep_chat_conversion_compatible(caller: dict | None) -> None:
    converter = ChatCompletionsConverter(return_token_id_information=False, custom_tool_strategy="function_bridge")
    omitted = NeMoGymResponseCreateParamsNonStreaming(input=CALLER_ITEMS)
    explicit_null = NeMoGymResponseCreateParamsNonStreaming(
        input=[{**item, "caller": caller} for item in CALLER_ITEMS]
    )
    assert converter.responses_to_chat_completion_create_params(omitted).model_dump() == (
        converter.responses_to_chat_completion_create_params(explicit_null).model_dump()
    )


@pytest.mark.parametrize("caller", [{"type": "direct"}, {"type": "program", "caller_id": "program_call_synthetic"}])
def test_function_call_and_result_preserve_caller_through_replay(caller: dict) -> None:
    items = [{**item, "caller": caller} for item in CALLER_ITEMS]
    serialized = _response(items).model_dump(mode="json", exclude_unset=True)["output"]
    assert serialized == items
    replay = NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": serialized})
    assert replay.model_dump(mode="json", exclude_unset=True)["input"] == items


@pytest.mark.parametrize("item", CALLER_ITEMS)
def test_program_caller_without_required_id_is_rejected(item: dict) -> None:
    malformed = item | {"caller": {"type": "program"}}
    with pytest.raises(ValidationError):
        _response([malformed])
    with pytest.raises(ValidationError):
        NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": [malformed]})


@pytest.mark.parametrize("mode", ["implicit", "explicit"])
def test_prompt_cache_options_survive_both_request_schemas_and_conversion(mode: str) -> None:
    options = {"mode": mode, "ttl": "30m"}
    chat = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(
        {"messages": [{"role": "user", "content": "synthetic input"}], "tools": [], "prompt_cache_options": options}
    )
    assert chat.model_dump(mode="json", exclude_unset=True)["prompt_cache_options"] == options
    converter = ChatCompletionsConverter(return_token_id_information=False)
    responses = converter.chat_completion_to_responses_create_params(chat)
    assert responses.model_dump(mode="json")["prompt_cache_options"] == options
    serialized = responses.model_dump(mode="json", exclude_unset=True)
    replay = NeMoGymResponseCreateParamsNonStreaming.model_validate(serialized)
    assert replay.prompt_cache_options == options
    assert converter.responses_to_chat_completion_create_params(replay).prompt_cache_options == options


@pytest.mark.parametrize("options", [{"mode": "invalid"}, {"ttl": "invalid"}])
def test_malformed_prompt_cache_options_remain_rejected(options: dict) -> None:
    with pytest.raises(ValidationError):
        NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": [], "prompt_cache_options": options})
    with pytest.raises(ValidationError):
        NeMoGymChatCompletionCreateParamsNonStreaming.model_validate({"messages": [], "prompt_cache_options": options})


def test_unknown_item_type_remains_rejected() -> None:
    unknown = {"type": "future_unknown_item", "id": "synthetic"}
    with pytest.raises(ValidationError):
        _response([unknown])
    with pytest.raises(ValidationError):
        NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": [unknown]})


def _anthropic_request(request: NeMoGymResponseCreateParamsNonStreaming) -> dict:
    return AnthropicConverter().responses_to_anthropic(
        request, model="synthetic", max_tokens=10, thinking=None, thinking_budget_tokens=None, extra_body={}
    )


@pytest.mark.parametrize("item", CALLER_ITEMS)
@pytest.mark.parametrize("caller", [{"type": "program", "caller_id": "sensitive-caller"}])
def test_anthropic_provider_request_rejects_caller_linkage(item: dict, caller: dict) -> None:
    request = NeMoGymResponseCreateParamsNonStreaming(input=[{**item, "caller": caller}])
    before = request.model_dump()
    with pytest.raises(NotImplementedError) as error:
        _anthropic_request(request)
    assert "sensitive-caller" not in str(error.value)
    assert request.model_dump() == before


@pytest.mark.parametrize("item", [FUNCTION_CALL, CUSTOM_CALL])
@pytest.mark.parametrize("caller", [{"type": "program", "caller_id": "sensitive-caller"}])
def test_anthropic_client_response_rejects_caller_linkage(item: dict, caller: dict) -> None:
    response = _response([{**item, "caller": caller}])
    before = response.model_dump()
    with pytest.raises(NotImplementedError) as error:
        AnthropicConverter().responses_to_anthropic_response(response, model="synthetic")
    assert "sensitive-caller" not in str(error.value)
    assert response.model_dump() == before


@pytest.mark.parametrize("caller", [None, {"type": "direct"}])
def test_anthropic_absent_null_and_direct_function_callers_remain_compatible(caller: dict | None) -> None:
    items = [FUNCTION_CALL, FUNCTION_OUTPUT]
    omitted = NeMoGymResponseCreateParamsNonStreaming(input=items)
    null = NeMoGymResponseCreateParamsNonStreaming(input=[{**item, "caller": caller} for item in items])
    assert _anthropic_request(omitted) == _anthropic_request(null)
    converter = AnthropicConverter()
    assert converter.responses_to_anthropic_response(_response([FUNCTION_CALL]), model="synthetic") == (
        converter.responses_to_anthropic_response(_response([{**FUNCTION_CALL, "caller": caller}]), model="synthetic")
    )


@pytest.mark.parametrize("tool_type", ["function", "custom"])
@pytest.mark.parametrize("allowed_callers", [[], ["direct"], ["programmatic"]])
@pytest.mark.parametrize("strategy", ["passthrough", "function_bridge"])
@pytest.mark.parametrize("drop", [False, True])
def test_chat_tool_conversion_rejects_caller_eligibility(
    tool_type: str, allowed_callers: list[str], strategy: str, drop: bool
) -> None:
    tool = {"type": tool_type, "name": "sensitive-tool", "allowed_callers": allowed_callers}
    if tool_type == "function":
        tool.update(parameters={}, strict=False)
    request = NeMoGymResponseCreateParamsNonStreaming(input="synthetic", tools=[tool])
    before = request.model_dump()
    converter = ChatCompletionsConverter(return_token_id_information=False, custom_tool_strategy=strategy)
    with pytest.raises(NotImplementedError, match="^Responses tool caller eligibility") as error:
        converter.responses_to_chat_completion_create_params(request, drop_unrepresentable_request_fields=drop)
    assert "sensitive-tool" not in str(error.value)
    assert request.model_dump() == before


@pytest.mark.parametrize("tool_type", ["function", "custom"])
@pytest.mark.parametrize("allowed_callers", [[], ["direct"], ["programmatic"]])
def test_anthropic_tool_conversion_rejects_caller_eligibility(tool_type: str, allowed_callers: list[str]) -> None:
    tool = {"type": tool_type, "name": "sensitive-tool", "allowed_callers": allowed_callers}
    if tool_type == "function":
        tool.update(parameters={}, strict=False)
    request = NeMoGymResponseCreateParamsNonStreaming(input="synthetic", tools=[tool])
    before = request.model_dump()
    with pytest.raises(NotImplementedError, match="^Responses tool caller eligibility") as error:
        _anthropic_request(request)
    assert "sensitive-tool" not in str(error.value)
    assert request.model_dump() == before
    assert NeMoGymResponseCreateParamsNonStreaming.model_validate(request.model_dump()).model_dump() == before


@pytest.mark.parametrize("tool_type", ["function", "custom"])
@pytest.mark.parametrize("strategy", ["passthrough", "function_bridge"])
def test_chat_tool_caller_eligibility_absent_and_null_remain_compatible(tool_type: str, strategy: str) -> None:
    tool = {"type": tool_type, "name": "synthetic-tool"}
    if tool_type == "function":
        tool.update(parameters={}, strict=False)
    converter = ChatCompletionsConverter(return_token_id_information=False, custom_tool_strategy=strategy)
    omitted = NeMoGymResponseCreateParamsNonStreaming(input="synthetic", tools=[tool])
    null = NeMoGymResponseCreateParamsNonStreaming(input="synthetic", tools=[tool | {"allowed_callers": None}])
    assert converter.responses_to_chat_completion_create_params(omitted).model_dump() == (
        converter.responses_to_chat_completion_create_params(null).model_dump()
    )


def test_anthropic_tool_caller_eligibility_absent_and_null_remain_compatible() -> None:
    tool = {"type": "function", "name": "synthetic-tool", "parameters": {}, "strict": False}
    omitted = NeMoGymResponseCreateParamsNonStreaming(input="synthetic", tools=[tool])
    null = NeMoGymResponseCreateParamsNonStreaming(input="synthetic", tools=[tool | {"allowed_callers": None}])
    assert _anthropic_request(omitted) == _anthropic_request(null)


@pytest.mark.parametrize("item", [CUSTOM_CALL, CUSTOM_OUTPUT])
def test_anthropic_null_custom_callers_keep_existing_unsupported_boundary(item: dict) -> None:
    errors = []
    for candidate in [item, item | {"caller": None}]:
        with pytest.raises(NotImplementedError) as error:
            _anthropic_request(NeMoGymResponseCreateParamsNonStreaming(input=[candidate]))
        errors.append(str(error.value))
    assert errors[0] == errors[1]


def test_anthropic_null_custom_response_caller_keeps_existing_unsupported_boundary() -> None:
    errors = []
    for candidate in [CUSTOM_CALL, CUSTOM_CALL | {"caller": None}]:
        with pytest.raises(NotImplementedError) as error:
            AnthropicConverter().responses_to_anthropic_response(_response([candidate]), model="synthetic")
        errors.append(str(error.value))
    assert errors[0] == errors[1]


@pytest.mark.parametrize("schema", [{}, {"type": "object", "properties": {"sensitive-field": {"type": "string"}}}])
def test_anthropic_rejects_function_output_schema_without_dropping_schema(schema: dict) -> None:
    tool = {"type": "function", "name": "synthetic", "parameters": {}, "strict": False, "output_schema": schema}
    request = NeMoGymResponseCreateParamsNonStreaming(input="synthetic", tools=[tool])
    before = request.model_dump(mode="json", exclude_unset=True)
    assert before["tools"] == [tool]
    assert (
        NeMoGymResponseCreateParamsNonStreaming.model_validate(before).model_dump(mode="json", exclude_unset=True)
        == before
    )
    with pytest.raises(NotImplementedError, match="^Responses tool output schema") as error:
        _anthropic_request(request)
    assert "sensitive-field" not in str(error.value)
    assert request.model_dump(mode="json", exclude_unset=True) == before


def test_anthropic_function_output_schema_absent_and_null_remain_compatible() -> None:
    tool = {"type": "function", "name": "synthetic", "parameters": {}, "strict": False}
    omitted = NeMoGymResponseCreateParamsNonStreaming(input="synthetic", tools=[tool])
    null = NeMoGymResponseCreateParamsNonStreaming(input="synthetic", tools=[tool | {"output_schema": None}])
    assert _anthropic_request(omitted) == _anthropic_request(null)


@pytest.mark.parametrize("schema", [{}, {"type": "object", "properties": {"sensitive-field": {"type": "string"}}}])
@pytest.mark.parametrize("drop", [False, True])
def test_chat_rejects_function_output_schema_without_echoing_payload(schema: dict, drop: bool) -> None:
    tool = {"type": "function", "name": "synthetic", "parameters": {}, "strict": False, "output_schema": schema}
    request = NeMoGymResponseCreateParamsNonStreaming(input="synthetic", tools=[tool])
    before = request.model_dump(mode="json", exclude_unset=True)
    converter = ChatCompletionsConverter(return_token_id_information=False)
    with pytest.raises(NotImplementedError, match="^Responses tool output schema") as error:
        converter.responses_to_chat_completion_create_params(request, drop_unrepresentable_request_fields=drop)
    assert "sensitive-field" not in str(error.value)
    assert request.model_dump(mode="json", exclude_unset=True) == before


def test_chat_function_output_schema_absent_and_null_remain_compatible() -> None:
    tool = {"type": "function", "name": "synthetic", "parameters": {}, "strict": False}
    converter = ChatCompletionsConverter(return_token_id_information=False)
    omitted = NeMoGymResponseCreateParamsNonStreaming(input="synthetic", tools=[tool])
    null = NeMoGymResponseCreateParamsNonStreaming(input="synthetic", tools=[tool | {"output_schema": None}])
    assert converter.responses_to_chat_completion_create_params(omitted).model_dump() == (
        converter.responses_to_chat_completion_create_params(null).model_dump()
    )
