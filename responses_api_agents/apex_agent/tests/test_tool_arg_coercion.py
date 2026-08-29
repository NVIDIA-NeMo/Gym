# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import json

import pytest
from pydantic import BaseModel, ValidationError

from responses_api_agents.apex_agent import stirrup_runtime


class DynamicModel(BaseModel):
    code: str | None = None


class WrappedParams(BaseModel):
    request: DynamicModel


def test_stringified_wrapper_value_is_unwrapped() -> None:
    arguments = json.dumps({"request": json.dumps({"code": "ls -la"})})

    coerced = stirrup_runtime.coerce_tool_arguments(WrappedParams, arguments)

    assert coerced is not None
    params = WrappedParams.model_validate_json(coerced)
    assert params.request.code == "ls -la"


def test_flat_arguments_are_wrapped_into_single_required_model_field() -> None:
    coerced = stirrup_runtime.coerce_tool_arguments(WrappedParams, json.dumps({"code": "ls -la"}))

    assert coerced is not None
    params = WrappedParams.model_validate_json(coerced)
    assert params.request.code == "ls -la"


def test_stringified_empty_object_is_unwrapped() -> None:
    coerced = stirrup_runtime.coerce_tool_arguments(WrappedParams, json.dumps({"request": "{}"}))

    assert coerced is not None
    params = WrappedParams.model_validate_json(coerced)
    assert params.request.code is None


def test_already_valid_arguments_are_left_untouched() -> None:
    arguments = json.dumps({"request": {"code": "ls -la"}})

    assert stirrup_runtime.coerce_tool_arguments(WrappedParams, arguments) is None


def test_string_typed_field_holding_json_text_is_not_mangled() -> None:
    class StringParams(BaseModel):
        s: str

    arguments = json.dumps({"s": json.dumps({"a": 1})})

    assert stirrup_runtime.coerce_tool_arguments(StringParams, arguments) is None
    assert StringParams.model_validate_json(arguments).s == '{"a": 1}'


def test_stringified_array_for_list_field_is_unwrapped() -> None:
    class PathsParams(BaseModel):
        paths: list[str]

    arguments = json.dumps({"paths": json.dumps(["article.txt", "chart.jpg"])})

    coerced = stirrup_runtime.coerce_tool_arguments(PathsParams, arguments)

    assert coerced is not None
    assert PathsParams.model_validate_json(coerced).paths == ["article.txt", "chart.jpg"]


def test_unfixable_arguments_return_none() -> None:
    assert stirrup_runtime.coerce_tool_arguments(WrappedParams, json.dumps({"request": "not json"})) is None
    assert stirrup_runtime.coerce_tool_arguments(WrappedParams, json.dumps({"request": 42})) is None
    assert stirrup_runtime.coerce_tool_arguments(WrappedParams, json.dumps([1, 2])) is None
    assert stirrup_runtime.coerce_tool_arguments(WrappedParams, "not json at all") is None


def test_empty_arguments_normalize_to_empty_object() -> None:
    # "{}" validates against an all-optional model, so no coercion is needed.
    assert stirrup_runtime.coerce_tool_arguments(DynamicModel, "   ") is None
    # A required plain-string field cannot be conjured from nothing.

    class NeedsString(BaseModel):
        s: str

    assert stirrup_runtime.coerce_tool_arguments(NeedsString, "") is None
    # A single required all-optional nested model wraps the normalized "{}".
    coerced = stirrup_runtime.coerce_tool_arguments(WrappedParams, "")
    assert coerced is not None
    assert WrappedParams.model_validate_json(coerced).request.code is None


def test_wrapping_does_not_fire_with_two_required_object_fields() -> None:
    class TwoRequired(BaseModel):
        first: DynamicModel
        second: DynamicModel

    assert stirrup_runtime.coerce_tool_arguments(TwoRequired, json.dumps({"code": "ls -la"})) is None


def test_unknown_keys_are_not_wrapped_into_an_all_optional_model() -> None:
    # pydantic's default extra="ignore" would validate the wrapped candidate while
    # dropping every emitted key; the repair must fall through to the error path.
    assert stirrup_runtime.coerce_tool_arguments(WrappedParams, json.dumps({"totally_wrong_field": 1})) is None
    assert (
        stirrup_runtime.coerce_tool_arguments(WrappedParams, json.dumps({"request": json.dumps({"bogus": 1})})) is None
    )


def test_structured_sibling_is_unwrapped_while_json_text_string_field_is_preserved() -> None:
    class MixedParams(BaseModel):
        request: DynamicModel
        notes: str

    arguments = json.dumps({"request": json.dumps({"code": "ls"}), "notes": json.dumps({"a": 1})})

    coerced = stirrup_runtime.coerce_tool_arguments(MixedParams, arguments)

    assert coerced is not None
    params = MixedParams.model_validate_json(coerced)
    assert params.request.code == "ls"
    assert params.notes == '{"a": 1}'


def test_never_raises_on_hostile_params_model() -> None:
    assert stirrup_runtime.coerce_tool_arguments(object(), json.dumps({"code": "ls"})) is None


def test_validation_error_formatting_names_field_and_previews_arguments() -> None:
    arguments = json.dumps({"request": "{}"})
    with pytest.raises(ValidationError) as exc_info:
        WrappedParams.model_validate_json(arguments)

    message = stirrup_runtime.format_tool_argument_validation_error(exc_info.value, arguments)

    assert message.startswith("Tool arguments are not valid: request: ")
    assert "(type=model_type)" in message
    assert "Submitted arguments (first 500 chars):" in message


def test_run_stirrup_rollout_installs_the_coercion_patch() -> None:
    # The sandbox runs stirrup_runtime.py standalone; the fix is inert unless
    # run_stirrup_rollout installs the patch before constructing the Agent.
    source = inspect.getsource(stirrup_runtime.run_stirrup_rollout)
    assert "install_tool_argument_coercion(Agent)" in source
    assert source.index("install_tool_argument_coercion(Agent)") < source.index("agent = Agent(")


async def test_patched_run_tool_coerces_mcp_arguments_but_never_finish_tools() -> None:
    pytest.importorskip("stirrup")
    from stirrup.core.agent import Agent
    from stirrup.core.models import Tool, ToolCall, ToolResult, ToolUseCountMetadata

    class FinishRequest(BaseModel):
        note: str | None = None

    class WrappedFinishParams(BaseModel):
        request: FinishRequest

    received: list[WrappedParams] = []

    async def executor(params: WrappedParams) -> ToolResult[ToolUseCountMetadata]:
        received.append(params)
        return ToolResult(content="ok", metadata=ToolUseCountMetadata())

    async def finish_executor(params: WrappedFinishParams) -> ToolResult[ToolUseCountMetadata]:
        return ToolResult(content="finished", metadata=ToolUseCountMetadata())

    class FakeClient:
        model_slug = "fake"
        max_tokens = 1024

        async def generate(self, messages, tools):  # pragma: no cover - never called
            raise NotImplementedError

    agent = Agent(
        client=FakeClient(),
        name="apex_coercion_test_agent",
        tools=[Tool(name="execute_code", description="Run code.", parameters=WrappedParams, executor=executor)],
        finish_tool=Tool(
            name="finish", description="Finish.", parameters=WrappedFinishParams, executor=finish_executor
        ),
    )

    unpatched_run_tool = Agent.run_tool
    try:
        stirrup_runtime.install_tool_argument_coercion(Agent)
        patched_run_tool = Agent.run_tool
        stirrup_runtime.install_tool_argument_coercion(Agent)
        assert Agent.run_tool is patched_run_tool

        shape_one_arguments = json.dumps({"request": json.dumps({"code": "ls -la"})})
        shape_one_call = ToolCall(name="execute_code", arguments=shape_one_arguments, tool_call_id="call_1")
        message = await agent.run_tool(shape_one_call, {})
        assert message.args_was_valid is True
        assert message.success is True
        assert [params.request.code for params in received] == ["ls -la"]
        # Coercion must not rewrite the model's raw emission in trajectory history.
        assert shape_one_call.arguments == shape_one_arguments

        # step() re-validates the ORIGINAL finish tool_call (stirrup agent.py:1261),
        # so finish arguments must stay unrepaired even when coercion could fix them.
        finish_message = await agent.run_tool(
            ToolCall(name="finish", arguments=json.dumps({"request": "{}"}), tool_call_id="call_2"), {}
        )
        assert finish_message.args_was_valid is False
        assert finish_message.content.startswith("Tool arguments are not valid: request: ")

        # Unrepairable arguments surface pydantic detail instead of the bare error.
        broken_message = await agent.run_tool(
            ToolCall(name="execute_code", arguments=json.dumps({"request": 42}), tool_call_id="call_3"), {}
        )
        assert broken_message.args_was_valid is False
        assert "Submitted arguments (first 500 chars):" in broken_message.content
    finally:
        Agent.run_tool = unpatched_run_tool
