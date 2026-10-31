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

import asyncio
import json
import os
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest
from nooa import Agent, strategy
from nooa.context_blocks.events import ResultStatus, ToolCallEvent, ToolResult
from nooa.context_blocks.formatter import MarkdownBlockFormatter
from nooa.context_blocks.render_config import RenderConfig
from nooa.context_blocks.renderer import render_context
from nooa.events import ExecutionResult, PythonOutput
from nooa.runtime.context_builder import _phase_events
from nooa.runtime.hooks import hooks_scope
from nooa.strategies.codeact_v2 import CodeActV2
from pydantic import BaseModel

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_collection import _build_trajectory_record
from nemo_gym.rollout_observability import AgentObservationBundle, TrajectoryRecord
from responses_api_agents.nooa_agent.gym_llm import (
    GymModelCall,
    GymResponsesLLM,
    PolicyCallBudgetExceeded,
    RolloutLLMState,
    _responses_input,
)
from responses_api_agents.nooa_agent.observability import (
    GymTraceHooks,
    _json_output,
    ensure_verifier_final_message,
    finalize_observation_gaps,
)
from responses_api_agents.nooa_agent.termination import NOOATerminationReason
from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, make_llm, model_response


@pytest.mark.asyncio
async def test_model_transport_preserves_scoped_ownership_on_failed_and_successful_attempts() -> None:
    from unittest.mock import AsyncMock, MagicMock

    trace = GymTraceHooks()
    state = RolloutLLMState(max_policy_calls=None)
    failure = ConnectionError("model disconnected")
    client = MagicMock(post=AsyncMock(side_effect=[failure, FakeHTTPResponse(model_response())]))
    llm = GymResponsesLLM(
        server_client=client,
        model_server_name="policy",
        model_url_path="/v1/responses",
        state=state,
        cookies={},
        on_call=trace.on_model_call,
    )
    root = trace.before_agent_call(call_id="parent", parent_call_id=None)
    child = trace.before_agent_call(call_id="child", parent_call_id="parent")
    try:
        with pytest.raises(ConnectionError, match="disconnected"):
            await llm.acall([{"role": "user", "content": "task"}])
    finally:
        trace.after_agent_call(context=child, exception=failure)
    try:
        await llm.acall([{"role": "user", "content": "task"}])
    finally:
        trace.after_agent_call(context=root, exception=None)
    assert [call.kwargs["headers"] for call in client.post.await_args_list] == [
        {"x-session-id": "child"},
        {"x-session-id": "parent"},
    ]
    assert [call.invocation_id for call in state.calls] == ["child", "parent"]
    assert state.calls[0].response is None
    assert state.calls[1].response is not None
    assert state.fatal_error is None


def test_scoped_projection_preserves_model_ownership_and_collector_tool_output_join() -> None:
    trace = GymTraceHooks()
    state = RolloutLLMState(max_policy_calls=1)
    context = trace.before_agent_call(call_id="root", parent_call_id=None)
    params = NeMoGymResponseCreateParamsNonStreaming(input="calculate")
    call = GymModelCall(
        model_ref=ModelServerRef(type="responses_api_models", name="policy"),
        request=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "calculate"}]),
    )
    trace.on_model_call(call)
    state.calls.append(call)
    call.response = NeMoGymResponse.model_validate(
        model_response(
            NeMoGymResponseFunctionToolCall(
                id="fc-1", call_id="call-1", name="execute_python", arguments='{"code":"print(7)"}'
            )
        )
    )
    execution = trace.before_code_execution(code="print(7)", execution_id="exec-1", tool_call_id="call-1")
    trace.after_code_execution(context=execution, result={"stdout": "7"}, exception=None)
    trace.after_agent_call(context=context, exception=None)
    episode, trajectory = trace.project(create_params=params, state=state, task_id="task", rollout_id="0-0")

    invocation = trajectory.invocations[0]
    assert invocation.invocation_id == "root"
    assert invocation.model_calls[0].response_id == "resp-1"
    assert [item.type for item in invocation.conversation] == ["message", "function_call", "function_call_output"]
    assert trajectory.turns[0].invocation_id == "root"
    assert trajectory.tool_calls[0].tool_call_id == "call-1"
    assert TrajectoryRecord.model_validate(trajectory.model_dump(mode="json")) == trajectory
    persisted = _build_trajectory_record(
        {"task_id": "task", "_ng_task_index": 0, "_ng_rollout_index": 0},
        {"ng_agent_observations": episode.observations.model_dump(mode="json")},
    )
    assert json.loads(persisted.tool_calls[0].output) == {"stdout": "7"}
    assert persisted.tool_calls[0].invocation_id == "root"


class NativeHistoryAgent(Agent):
    @strategy(CodeActV2())
    async def answer(self, question: str) -> str:
        """Answer the question using Python."""
        ...


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_cell", [False, True])
async def test_native_codeact_history_keeps_prefills_and_actual_cell_feedback(failed_cell: bool) -> None:
    trace = GymTraceHooks()
    state = RolloutLLMState(max_policy_calls=2)
    code = "print('cell marker')"
    if failed_cell:
        code += "\nraise ValueError('controlled bad cell')"
    responses = [
        model_response(
            NeMoGymResponseFunctionToolCall(
                id="fc-1", call_id="call-1", name="python_cell", arguments=json.dumps({"code": code})
            )
        ),
        model_response(
            NeMoGymResponseFunctionToolCall(
                id="fc-final",
                call_id="call-final",
                name="python_cell",
                arguments=json.dumps({"code": "print('final marker')\nreturn_result('done')"}),
            ),
            response_id="resp-final",
        ),
    ]
    client = MagicMock(post=AsyncMock(side_effect=[FakeHTTPResponse(response) for response in responses]))
    llm = GymResponsesLLM(
        server_client=client,
        model_server_name="policy",
        model_url_path="/v1/responses",
        state=state,
        cookies={},
        on_call=trace.on_model_call,
    )
    agent = NativeHistoryAgent(llm=llm)
    try:
        with hooks_scope(trace):
            assert await agent.answer("test the trace") == "done"
        before = [(call.request.model_dump_json(), call.response.model_dump_json()) for call in state.calls]
        episode, trajectory = trace.project(
            create_params=NeMoGymResponseCreateParamsNonStreaming(input="test the trace"),
            state=state,
            task_id="task",
            rollout_id="rollout",
        )
        assert before == [(call.request.model_dump_json(), call.response.model_dump_json()) for call in state.calls]

        native_calls = {event.tool_call_id: event for event in agent.events.query(type="ToolCallEvent")}
        native_outputs = {event.tool_call_id: event for event in agent.events.query(type="PythonOutput")}
        prefill_ids = {call_id for call_id, event in native_calls.items() if event.metadata.get("prefill")}
        tools = {tool.tool_call_id: tool for tool in trajectory.tool_calls}
        assert prefill_ids
        assert tools.keys() == native_outputs.keys() == prefill_ids | {"call-1", "call-final"}
        for call_id, tool in tools.items():
            assert tool.tool_name == native_calls[call_id].name == "python_cell"
        assert tools["call-1"].status == ("failed" if failed_cell else "completed")
        for call_id in prefill_ids | {"call-1"}:
            visible = next(
                item.content
                for call in state.calls
                for item in call.request.input
                if getattr(item, "role", None) == "user"
                and isinstance(item.content, str)
                and item.content.startswith(f'<sys tag="{native_outputs[call_id].tag}">')
            )
            assert tools[call_id].output == visible
            assert any(
                isinstance(item, NeMoGymFunctionCallOutput)
                and item.call_id == call_id
                and item.output == "status: accepted"
                for call in state.calls
                for item in call.request.input
            )
        assert "cell marker" in tools["call-1"].output
        if failed_cell:
            assert "controlled bad cell" in tools["call-1"].output
        # The final cell never reaches another request: preserve its captured result.
        assert json.loads(tools["call-final"].output)["stdout"] == "final marker\n"
        assert [gap.code for gap in trajectory.gaps] == ["public_assistant_metadata_unavailable"]
        persisted = _build_trajectory_record(
            {"task_id": "task", "_ng_task_index": 0, "_ng_rollout_index": 0},
            {"ng_agent_observations": episode.observations.model_dump(mode="json")},
        )
        assert {tool.tool_call_id: tool.output for tool in persisted.tool_calls} == {
            call_id: tool.output for call_id, tool in tools.items()
        }
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("capture_rerender", [False, True])
@pytest.mark.parametrize("change_format", [False, True])
async def test_native_feedback_keeps_latest_captured_render_after_mutation(
    capture_rerender: bool, change_format: bool
) -> None:
    trace = GymTraceHooks()
    budget = 3 if capture_rerender else 2
    state = RolloutLLMState(max_policy_calls=budget)
    codes = ["values = ['first']\nvalues", "values.append('second')\nprint('mutated')"]
    if change_format:
        codes = [
            "values = {'long_value': 'x' * 100}\nvalues",
            "for event in self.events.query(type='PythonOutput'):\n"
            "    if event.tool_call_id == 'call-1':\n"
            "        event.metadata['truncation_event_format'] = {'max_string': 10}\n"
            "print('format updated')",
        ]
    if capture_rerender:
        codes.append("return_result('done')")
    client = MagicMock(
        post=AsyncMock(
            side_effect=[
                FakeHTTPResponse(
                    model_response(
                        NeMoGymResponseFunctionToolCall(
                            id=f"fc-{index}",
                            call_id=f"call-{index}",
                            name="python_cell",
                            arguments=json.dumps({"code": code}),
                        ),
                        response_id=f"resp-{index}",
                    )
                )
                for index, code in enumerate(codes, 1)
            ]
        )
    )
    llm = GymResponsesLLM(
        server_client=client,
        model_server_name="policy",
        model_url_path="/v1/responses",
        state=state,
        cookies={},
        on_call=trace.on_model_call,
    )
    agent = NativeHistoryAgent(llm=llm)
    try:
        with hooks_scope(trace):
            if capture_rerender:
                assert await agent.answer("test mutable feedback") == "done"
            else:
                with pytest.raises(Exception, match="policy call budget exhausted") as caught:
                    await agent.answer("test mutable feedback")
                assert isinstance(caught.value.__cause__, PolicyCallBudgetExceeded)
        assert len(state.calls) == client.post.await_count == state.used == budget
        before = [call.request.model_dump_json() for call in state.calls]
        native_output = next(
            event for event in agent.events.query(type="PythonOutput") if event.tool_call_id == "call-1"
        )
        captured = [
            item.content
            for call in state.calls
            for item in call.request.input
            if getattr(item, "role", None) == "user"
            and isinstance(item.content, str)
            and item.content.startswith(f'<sys tag="{native_output.tag}">')
        ]
        assert len(captured) == budget - 1
        if capture_rerender:
            assert captured[0] != captured[1]
        elif change_format:
            assert native_output.metadata["truncation_event_format"] == {"max_string": 10}
            assert "x" * 100 in captured[0]
        else:
            assert native_output.value == ["first", "second"]
            assert "second" not in captured[0]
        episode, trajectory = trace.project(
            create_params=NeMoGymResponseCreateParamsNonStreaming(input="test mutable feedback"),
            state=state,
            task_id="task",
            rollout_id="rollout",
        )
        assert before == [call.request.model_dump_json() for call in state.calls]
        assert next(tool.output for tool in trajectory.tool_calls if tool.tool_call_id == "call-1") == captured[-1]
        assert [gap.code for gap in trajectory.gaps] == ["public_assistant_metadata_unavailable"]
        persisted = _build_trajectory_record(
            {"task_id": "task", "_ng_task_index": 0, "_ng_rollout_index": 0},
            {"ng_agent_observations": episode.observations.model_dump(mode="json")},
        )
        assert next(tool.output for tool in persisted.tool_calls if tool.tool_call_id == "call-1") == captured[-1]
    finally:
        await agent.aclose()


def _native_messages(agent: Agent) -> list[object]:
    return render_context(
        _phase_events([], agent.event_manager),
        block_formatter=agent.render_config.block_formatter,
        provider_formatter=agent.render_config.provider_formatter,
        event_format_resolver=agent.runtime._event_format_for_event,
    ).output


def _record_native_cell(
    trace: GymTraceHooks,
    agent: Agent,
    *,
    invocation_id: str,
    call_id: str,
    stdout: str,
    event_format: dict | None = None,
) -> PythonOutput:
    metadata = {"call_id": invocation_id}
    if event_format is not None:
        metadata["truncation_event_format"] = event_format
    agent.event_manager.add(
        ToolCallEvent(
            tool_call_id=call_id,
            name="python_cell",
            arguments={"code": f"print({stdout!r})"},
            result=ToolResult(tool_call_id=call_id, content="status: accepted"),
            metadata=metadata,
        )
    )
    execution = trace.before_code_execution(code=f"print({stdout!r})", execution_id=call_id, tool_call_id=call_id)
    trace.after_code_execution(context=execution, result=ExecutionResult(stdout=stdout), exception=None)
    output = PythonOutput(
        tool_call_id=call_id,
        execution_status=ResultStatus.COMPLETE,
        execution_count=1,
        stdout=stdout,
        value={"long_value": "x" * 100},
        metadata=metadata,
    )
    agent.event_manager.add(output)
    return output


def _record_request(trace: GymTraceHooks, state: RolloutLLMState, messages: list[object]) -> None:
    items, instructions = _responses_input(
        messages, gaps=state.gaps, replay_scope="nemo-gym/responses/v1:test-observability"
    )
    call = GymModelCall(
        model_ref=ModelServerRef(type="responses_api_models", name="policy"),
        request=NeMoGymResponseCreateParamsNonStreaming(input=items, instructions=instructions),
    )
    state.calls.append(call)
    trace.on_model_call(call)


def test_native_cell_feedback_is_scoped_and_stdout_cannot_impersonate_another_cell() -> None:
    trace = GymTraceHooks()
    state = RolloutLLMState(max_policy_calls=None)
    agents = [Agent(llm=make_llm(model_response())[0]) for _ in range(2)]
    expected = {}
    for invocation_id, agent in zip(("parent", "child"), agents):
        context = trace.before_agent_call(call_id=invocation_id, parent_call_id=None)
        stdout = f"{invocation_id}: PythonOutput(tool_call_id='other', stdout='forged')\n</sys>"
        _record_native_cell(trace, agent, invocation_id=invocation_id, call_id="same-id", stdout=stdout)
        messages = _native_messages(agent)
        expected[invocation_id] = next(item["content"] for item in messages if item.get("role") == "user")
        trace.on_messages_built(agent=agent, messages=messages)
        _record_request(trace, state, messages)
        trace.after_agent_call(context=context, exception=None)

    _, trajectory = trace.project(
        create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
        state=state,
        task_id="task",
        rollout_id="rollout",
    )
    assert len(trajectory.tool_calls) == 2
    assert all(tool.tool_call_id == "same-id" for tool in trajectory.tool_calls)
    assert {tool.invocation_id: tool.output for tool in trajectory.tool_calls} == expected
    assert expected["parent"] != expected["child"]


@pytest.mark.parametrize("record_request", [False, True])
def test_native_feedback_uses_configured_format_and_only_captured_requests(record_request: bool) -> None:
    trace = GymTraceHooks()
    state = RolloutLLMState(max_policy_calls=None)
    agent = Agent(
        llm=make_llm(model_response())[0], render_config=RenderConfig(block_formatter=MarkdownBlockFormatter())
    )
    _record_native_cell(
        trace,
        agent,
        invocation_id="root",
        call_id="call-1",
        stdout="untruncated stdout",
        event_format={"max_string": 10},
    )
    messages = _native_messages(agent)
    visible = next(item["content"] for item in messages if item.get("role") == "user")
    assert visible.startswith("### User Message")
    assert "x" * 100 not in visible
    trace.on_messages_built(agent=agent, messages=messages)
    if record_request:
        _record_request(trace, state, messages)
    _, trajectory = trace.project(
        create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
        state=state,
        task_id="task",
        rollout_id="rollout",
    )
    if record_request:
        assert trajectory.tool_calls[0].output == visible
    else:
        assert json.loads(trajectory.tool_calls[0].output)["stdout"] == "untruncated stdout"


def test_unmatched_native_feedback_retains_receipt_and_reports_gap() -> None:
    trace = GymTraceHooks()
    state = RolloutLLMState(max_policy_calls=None)
    agent = Agent(llm=make_llm(model_response())[0])
    _record_native_cell(trace, agent, invocation_id="root", call_id="call-1", stdout="actual stdout")
    messages = _native_messages(agent)
    # Simulate an additional context-dependent renderer transformation. The
    # producer must not claim its independently rendered candidate was observed.
    for message in messages:
        if message.get("role") == "user":
            message["content"] = "custom transformed feedback"
    trace.on_messages_built(agent=agent, messages=messages)
    _record_request(trace, state, messages)
    _, trajectory = trace.project(
        create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
        state=state,
        task_id="task",
        rollout_id="rollout",
    )
    assert trajectory.tool_calls[0].output == "status: accepted"
    assert [gap.code for gap in trajectory.gaps] == [
        "public_assistant_metadata_unavailable",
        "python_output_projection_unavailable",
    ]


def test_later_context_filtering_does_not_erase_verified_native_feedback() -> None:
    trace = GymTraceHooks()
    state = RolloutLLMState(max_policy_calls=None)
    agent = Agent(llm=make_llm(model_response())[0])
    _record_native_cell(trace, agent, invocation_id="root", call_id="call-1", stdout="verified stdout")
    messages = _native_messages(agent)
    visible = next(item["content"] for item in messages if item.get("role") == "user")
    trace.on_messages_built(agent=agent, messages=messages)
    _record_request(trace, state, messages)
    # A later request no longer includes the old event (for example after
    # compaction). Its earlier model-visible output remains authoritative.
    trace.on_messages_built(agent=agent, messages=[])
    _record_request(trace, state, [])
    _, trajectory = trace.project(
        create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
        state=state,
        task_id="task",
        rollout_id="rollout",
    )
    assert trajectory.tool_calls[0].output == visible
    assert [gap.code for gap in trajectory.gaps] == ["public_assistant_metadata_unavailable"]


def test_uncaptured_return_to_an_older_render_does_not_replace_latest_capture() -> None:
    trace = GymTraceHooks()
    state = RolloutLLMState(max_policy_calls=None)
    agent = Agent(llm=make_llm(model_response())[0])
    output = _record_native_cell(trace, agent, invocation_id="root", call_id="call-1", stdout="feedback")
    captured = []
    for index, event_format in enumerate(({}, {"max_string": 10}, {})):
        output.metadata["truncation_event_format"] = event_format
        messages = _native_messages(agent)
        trace.on_messages_built(agent=agent, messages=messages)
        if index < 2:
            _record_request(trace, state, messages)
            captured.append(next(item["content"] for item in messages if item.get("role") == "user"))
    assert captured[0] != captured[1]
    assert next(item["content"] for item in messages if item.get("role") == "user") == captured[0]
    _, trajectory = trace.project(
        create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
        state=state,
        task_id="task",
        rollout_id="rollout",
    )
    assert trajectory.tool_calls[0].output == captured[-1]
    assert [gap.code for gap in trajectory.gaps] == ["public_assistant_metadata_unavailable"]


@pytest.mark.parametrize("has_later_model_call", [False, True])
def test_code_execution_module_locals_do_not_prevent_trace_projection(has_later_model_call: bool) -> None:
    trace = GymTraceHooks()
    state = RolloutLLMState(max_policy_calls=1)
    params = NeMoGymResponseCreateParamsNonStreaming(input="task")
    execution = trace.before_code_execution(
        code="import os\nprint(os.name)", execution_id="exec-1", tool_call_id="call-1"
    )
    raw_result = ExecutionResult(stdout="posix\n", captured_locals={"os": os})
    trace.after_code_execution(context=execution, result=raw_result, exception=None)
    visible_output = "Stdout:\nposix\n"
    if has_later_model_call:
        call = GymModelCall(
            model_ref=ModelServerRef(type="responses_api_models", name="policy"),
            request=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymFunctionCallOutput(call_id="call-1", output=visible_output)]
            ),
        )
        trace.on_model_call(call)
        state.calls.append(call)

    episode, trajectory = trace.project(create_params=params, state=state, task_id="task", rollout_id="0-0")

    tool = trajectory.tool_calls[0]
    assert tool is not execution.observation
    if has_later_model_call:
        assert tool.output == visible_output
    else:
        output = json.loads(tool.output)
        assert output["stdout"] == "posix\n"
        assert "captured_locals" not in output
    assert episode.response.output[-1].output == tool.output
    assert NeMoGymResponse.model_validate_json(episode.response.model_dump_json()) == episode.response
    assert AgentObservationBundle.model_validate_json(episode.observations.model_dump_json()) == episode.observations
    assert TrajectoryRecord.model_validate_json(trajectory.model_dump_json()) == trajectory
    snapshot = json.loads(execution.observation.output)
    assert snapshot["stdout"] == "posix\n"
    assert "captured_locals" not in snapshot
    assert raw_result.captured_locals["os"] is os


def test_no_model_call_preserves_original_responses_input_with_tool_events() -> None:
    trace = GymTraceHooks()
    params = NeMoGymResponseCreateParamsNonStreaming(input="original row input")
    with trace.resource_call("lookup", {"query": "q"}) as observation:
        observation.output = {"result": "found"}

    episode, trajectory = trace.project(
        create_params=params,
        state=RolloutLLMState(max_policy_calls=1),
        task_id="task",
        rollout_id="0-0",
    )

    invocation = trajectory.invocations[0]
    assert invocation.invocation_id == "root"
    assert invocation.conversation[0].content == "original row input"
    assert [item.type for item in invocation.conversation[1:]] == ["function_call", "function_call_output"]
    assert trajectory.turns == []


def test_code_output_snapshots_imported_modules_and_mutable_locals() -> None:
    from nooa.events import ExecutionResult

    trace = GymTraceHooks()
    execution = trace.before_code_execution(code="import math", execution_id="exec-1", tool_call_id="call-1")
    variables = {"math": ModuleType("math"), "values": [1]}
    result = ExecutionResult(stdout="done", captured_locals=variables, returned_value=variables["values"])
    trace.after_code_execution(context=execution, result=result, exception=None)
    variables["values"].append(2)

    episode, trajectory = trace.project(
        create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
        state=RolloutLLMState(max_policy_calls=1),
        task_id="task",
        rollout_id="0-0",
    )

    output = json.loads(trajectory.tool_calls[0].output)
    assert output["stdout"] == "done"
    assert output["returned_value"] == [1]
    assert "captured_locals" not in output
    assert trajectory.tool_calls[0].status == "completed"
    assert episode.response.output[-1].output == trajectory.tool_calls[0].output
    TrajectoryRecord.model_validate_json(trajectory.model_dump_json())


@pytest.mark.parametrize("error", [ValueError("bad code"), asyncio.CancelledError()])
def test_failed_and_cancelled_code_keeps_matching_output(error: BaseException) -> None:
    trace = GymTraceHooks()
    context = trace.before_agent_call(call_id="root", parent_call_id=None)
    execution = trace.before_code_execution(code="raise ValueError()", execution_id="exec-1", tool_call_id="call-1")
    trace.after_code_execution(context=execution, result=None, exception=error)
    trace.after_agent_call(context=context, exception=error)
    episode, trajectory = trace.project(
        create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
        state=RolloutLLMState(max_policy_calls=1),
        task_id="task",
        rollout_id="0-0",
    )
    assert trajectory.tool_calls[0].status == ("cancelled" if isinstance(error, asyncio.CancelledError) else "failed")
    assert trajectory.invocations[0].status == "failed"
    output = episode.response.output[-1]
    assert output.call_id == trajectory.tool_calls[0].tool_call_id
    assert json.loads(output.output)["error_type"] == type(error).__name__


def test_final_projection_preserves_output_for_a_tool_without_an_after_hook() -> None:
    trace = GymTraceHooks()
    context = trace.before_agent_call(call_id="root", parent_call_id=None)
    trace.before_code_execution(code="await pending()", execution_id="exec-1", tool_call_id="call-1")
    trace.after_agent_call(context=context, exception=None)
    episode, trajectory = trace.project(
        create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
        state=RolloutLLMState(max_policy_calls=1),
        task_id="task",
        rollout_id="0-0",
    )
    assert [item.type for item in episode.response.output] == ["function_call", "function_call_output"]
    output = episode.response.output[-1]
    assert output.call_id == "call-1"
    assert output.status == "incomplete"
    assert json.loads(output.output) is None
    assert trajectory.tool_calls[0].status == "incomplete"
    assert trajectory.tool_calls[0].completed_at is None
    assert trajectory.invocations[0].conversation[-1] == output


@pytest.mark.asyncio
async def test_parallel_tool_completions_checkpoint_only_finished_outputs() -> None:
    snapshots = []

    def checkpoint() -> None:
        episode, trajectory = trace.project(
            create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
            state=RolloutLLMState(max_policy_calls=None),
            task_id="task",
            rollout_id="rollout",
            partial=True,
        )
        # Serialize inside the callback while the event state is stable.
        snapshots.append((episode.response.model_dump(mode="json"), trajectory.model_dump(mode="json")))

    trace = GymTraceHooks(on_progress=checkpoint)
    first_done = asyncio.Event()
    both_started = asyncio.Event()

    async def first() -> None:
        call = trace.before_tool_execution(tool_name="first", arguments={}, execution_id="first")
        await both_started.wait()
        trace.after_tool_execution(context=call, result={"value": 1}, exception=None)
        first_done.set()

    async def second() -> None:
        call = trace.before_tool_execution(tool_name="second", arguments={}, execution_id="second")
        both_started.set()
        await first_done.wait()
        trace.after_tool_execution(context=call, result=None, exception=ValueError("tool failed"))

    await asyncio.gather(first(), second())
    assert len(snapshots) == 2
    assert [tool["status"] for tool in snapshots[0][1]["tool_calls"]] == ["completed", "incomplete"]
    assert [item["call_id"] for item in snapshots[0][0]["output"] if item["type"] == "function_call_output"] == [
        "first"
    ]
    assert [tool["status"] for tool in snapshots[1][1]["tool_calls"]] == ["completed", "failed"]
    assert json.loads(snapshots[1][1]["tool_calls"][1]["output"])["error_type"] == "ValueError"


class TypedAnswer(BaseModel):
    value: int


@pytest.mark.parametrize("missing_usage", [False, True])
@pytest.mark.parametrize("failed_attempt", [False, True])
def test_episode_usage_sums_all_models_only_when_complete(missing_usage: bool, failed_attempt: bool) -> None:
    trace = GymTraceHooks()
    state = RolloutLLMState(max_policy_calls=2)
    params = NeMoGymResponseCreateParamsNonStreaming(input=[])
    for index in range(2):
        response = NeMoGymResponse.model_validate(model_response(response_id=f"r-{index}"))
        response.usage = (
            None
            if missing_usage and index == 1
            else NeMoGymResponseUsage.model_validate(
                {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                    "input_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 1},
                }
            )
        )
        call = GymModelCall(
            model_ref=ModelServerRef(type="responses_api_models", name=f"model-{index}"),
            request=params,
            response=response,
        )
        trace.on_model_call(call)
        state.calls.append(call)
    if failed_attempt:
        failed = GymModelCall(model_ref=ModelServerRef(type="responses_api_models", name="model-0"), request=params)
        trace.on_model_call(failed)
        state.calls.append(failed)
    episode, _ = trace.project(create_params=params, state=state, task_id="t", rollout_id="r")
    if missing_usage:
        assert episode.response.usage is None
    else:
        usage = episode.response.usage
        assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (20, 4, 24)
        assert usage.input_tokens_details.cached_tokens == 6
        assert usage.output_tokens_details.reasoning_tokens == 2


@pytest.mark.parametrize("attempts", [0, 2])
def test_no_completed_model_responses_have_known_zero_aggregate_usage(attempts: int) -> None:
    trace = GymTraceHooks()
    state = RolloutLLMState(max_policy_calls=None)
    params = NeMoGymResponseCreateParamsNonStreaming(input="rejected task")
    for _ in range(attempts):
        call = GymModelCall(model_ref=ModelServerRef(type="responses_api_models", name="policy"), request=params)
        trace.on_model_call(call)
        state.calls.append(call)
    episode, trajectory = trace.project(create_params=params, state=state, task_id="t", rollout_id="r")
    assert episode.response.usage.input_tokens == 0
    assert episode.response.usage.output_tokens == 0
    assert episode.response.usage.total_tokens == 0
    assert all(call.response is None for call in state.calls)
    assert trajectory.turns == []


def test_failed_resource_and_pending_model_keep_evidence() -> None:
    trace = GymTraceHooks()
    state = RolloutLLMState(max_policy_calls=1)
    context = trace.before_agent_call(call_id="outer", parent_call_id=None)
    params = NeMoGymResponseCreateParamsNonStreaming(input="question", instructions="system")
    call = GymModelCall(model_ref=ModelServerRef(type="responses_api_models", name="policy"), request=params)
    state.calls.append(call)
    with trace.activate_agent_call(context):
        trace.on_model_call(call)
        with pytest.raises(ConnectionError):
            with trace.resource_call("lookup", {"query": "q"}):
                raise ConnectionError("unavailable")
    trace.after_agent_call(context=context, exception=ConnectionError())
    episode, trajectory = trace.project(create_params=params, state=state, task_id="t", rollout_id="r")
    assert trajectory.turns == []
    assert trajectory.tool_calls[0].status == "failed"
    assert trajectory.tool_calls[0].error_type == "ConnectionError"
    assert episode.observations.records[0].conversation[0].content == "system"
    assert episode.observations.records[0].invocation_id == "outer"


@pytest.mark.parametrize("value", [TypedAnswer(value=7), {"nested": [TypedAnswer(value=7)]}, 0, False, None])
def test_terminal_values_are_json_not_python_representations(value: object) -> None:
    normalized = json.loads(_json_output(value))
    if isinstance(value, TypedAnswer):
        assert normalized == {"value": 7}
    elif isinstance(value, dict):
        assert normalized == {"nested": [{"value": 7}]}
    else:
        assert normalized == value


def test_ensure_verifier_final_message_adds_fallback_without_mutating_episode() -> None:
    response = NeMoGymResponse(
        id="nooa-test",
        created_at=0,
        model="nooa",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )

    adapted, gaps = ensure_verifier_final_message(response, "fallback answer")

    assert adapted.output[0].content[0].text == "fallback answer"
    assert [gap.code for gap in gaps] == ["non_trainable_terminal_output"]
    assert response.output == []


def test_ensure_verifier_final_message_appends_after_intermediate_message_and_tool_call() -> None:
    response = NeMoGymResponse(
        id="nooa-test",
        created_at=0,
        model="nooa",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="intermediate",
                content=[NeMoGymResponseOutputText(annotations=[], text="I will check.")],
            ),
            NeMoGymResponseFunctionToolCall(
                id="return-1",
                call_id="return-1",
                name="return_result",
                arguments='{"result":"cold"}',
            ),
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )

    adapted, gaps = ensure_verifier_final_message(response, "It is cold.")

    assert [item.type for item in adapted.output] == ["message", "function_call", "message"]
    assert adapted.output[-1].content[0].text == "It is cold."
    assert [gap.code for gap in gaps] == ["non_trainable_terminal_output"]


def test_ensure_verifier_final_message_preserves_terminal_message() -> None:
    response = NeMoGymResponse(
        id="nooa-test",
        created_at=0,
        model="nooa",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="final",
                content=[NeMoGymResponseOutputText(annotations=[], text="It is cold.")],
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )

    adapted, gaps = ensure_verifier_final_message(response, "It is cold.")

    assert adapted is response
    assert gaps == []


def test_typed_entrypoint_return_is_not_replaced_by_a_childs_last_message() -> None:
    response = NeMoGymResponse.model_validate(
        model_response(
            NeMoGymResponseOutputMessage(
                id="child", content=[NeMoGymResponseOutputText(annotations=[], text="intermediate child answer")]
            )
        )
    )
    adapted, gaps = ensure_verifier_final_message(response, {"answer": TypedAnswer(value=7)})
    assert json.loads(adapted.output[-1].content[0].text) == {"answer": {"value": 7}}
    assert len(response.output) == 1
    assert gaps[0].code == "non_trainable_terminal_output"


def test_finalize_observation_gaps_appends_termination_gap() -> None:
    bundle = AgentObservationBundle(source="nooa", records=[], gaps=[])

    finalized = finalize_observation_gaps(
        bundle,
        termination_reason=NOOATerminationReason.POLICY_BUDGET_EXCEEDED,
        termination_error="budget exhausted",
    )

    assert finalized.gaps[0].code == "policy_budget_exceeded"
    assert finalized.gaps[0].detail == "budget exhausted"


@pytest.mark.parametrize("invalid_output", [False, True])
@pytest.mark.parametrize("outer_failure", [False, True])
def test_policy_stop_labels_preserve_unrelated_outer_failures(invalid_output: bool, outer_failure: bool) -> None:
    from responses_api_agents.nooa_agent.gym_llm import InvalidPolicyOutputError, PolicyCallBudgetExceeded

    stop = InvalidPolicyOutputError("invalid answer") if invalid_output else PolicyCallBudgetExceeded("call limit")
    error = RuntimeError("cleanup failed") if outer_failure else stop
    if outer_failure:
        error.__cause__ = stop
    trace = GymTraceHooks()
    context = trace.before_agent_call(call_id="root", parent_call_id=None)
    trace.after_agent_call(context=context, exception=error)
    _, trajectory = trace.project(
        create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
        state=RolloutLLMState(max_policy_calls=1),
        task_id="task",
        rollout_id="0-0",
    )
    assert trajectory.invocations[0].status == ("failed" if outer_failure else "incomplete")
    assert trajectory.invocations[0].error_type == type(error).__name__


@pytest.mark.parametrize("invalid_output", [False, True])
def test_final_root_outcome_preserves_child_and_checkpoint_outcomes(invalid_output: bool) -> None:
    from responses_api_agents.nooa_agent.gym_llm import InvalidPolicyOutputError, PolicyCallBudgetExceeded

    stop = InvalidPolicyOutputError("invalid answer") if invalid_output else PolicyCallBudgetExceeded("call limit")
    trace = GymTraceHooks()
    context = trace.before_agent_call(call_id="root", parent_call_id=None)
    child = trace.before_agent_call(call_id="child", parent_call_id="root")
    trace.after_agent_call(context=child, exception=stop)
    trace.after_agent_call(context=context, exception=stop)
    summary = trace.before_agent_call(call_id="summary", parent_call_id=None)
    trace.after_agent_call(context=summary, exception=None)
    projection = dict(
        create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
        state=RolloutLLMState(max_policy_calls=1),
        task_id="task",
        rollout_id="0-0",
    )
    _, checkpoint = trace.project(**projection)
    assert checkpoint.invocations[0].status == "incomplete"
    # The invocation adapter finishes cleanup after NOOA emits after_agent_call.
    stop.nooa_cleanup_errors = (ConnectionError("cleanup transport failed"),)
    _, final = trace.project(**projection, root_status="failed", root_error_type=type(stop).__name__)
    assert final.invocations[0].status == "failed"
    assert final.invocations[0].error_type == type(stop).__name__
    assert checkpoint.invocations[0].status == "incomplete"
    assert final.invocations[1].status == checkpoint.invocations[1].status == "incomplete"
    assert final.invocations[2].status == checkpoint.invocations[2].status == "completed"
    # Projection never revisits the shared, now mutated exception or its frames.
    _, unchanged = trace.project(**projection)
    assert [invocation.status for invocation in unchanged.invocations] == ["incomplete", "incomplete", "completed"]


def test_delegated_sessions_link_to_the_invocation_that_opened_the_parent_session(monkeypatch) -> None:
    from responses_api_agents.nooa_agent import observability

    sessions = iter([("root-session", None), ("child-session", "root-session"), ("grandchild", "child-session")])
    monkeypatch.setattr(observability, "_current_atom_session", lambda: next(sessions))
    trace = GymTraceHooks()
    root = trace.before_agent_call(call_id="call-root", parent_call_id=None)
    # NOOA reports no traced caller for a delegated session's own method calls.
    child = trace.before_agent_call(call_id="call-child", parent_call_id=None)
    grandchild = trace.before_agent_call(call_id="call-grandchild", parent_call_id=None)
    for context in (grandchild, child, root):
        trace.after_agent_call(context=context, exception=None)

    invocations = trace._invocations
    assert invocations["call-root"].parent_invocation_id is None
    assert invocations["call-root"].harness_session_id == "root-session"
    assert invocations["call-child"].parent_invocation_id == "call-root"
    assert invocations["call-child"].harness_session_id == "child-session"
    assert invocations["call-grandchild"].parent_invocation_id == "call-child"


def test_without_atom_session_context_the_reported_parent_call_is_kept() -> None:
    trace = GymTraceHooks()
    root = trace.before_agent_call(call_id="root", parent_call_id=None)
    child = trace.before_agent_call(call_id="child", parent_call_id="root")
    trace.after_agent_call(context=child, exception=None)
    trace.after_agent_call(context=root, exception=None)
    assert trace._invocations["child"].parent_invocation_id == "root"


def test_the_root_atom_session_is_parented_to_the_trace_session_that_opened_it(monkeypatch) -> None:
    pytest.importorskip("nooa.tracing._session")
    from nooa.tracing import _session
    from opentelemetry import context as otel_context

    from responses_api_agents.nooa_agent import observability

    attributes: dict[str, str] = {}
    monkeypatch.setattr(_session, "get_span_attributes", lambda context=None: attributes)
    monkeypatch.setattr(
        otel_context, "get_value", lambda key, context=None: "trace-1" if key == "session.id" else None
    )

    # Entry calls before Atom opens a session: the trace session stands in.
    assert observability._current_atom_session() == ("trace-1", None)
    # The root Atom session (no Atom parent) hangs off the trace session.
    attributes.update({observability.ATOM_SESSION_ID_ATTR: "root-atom"})
    assert observability._current_atom_session() == ("root-atom", "trace-1")
    # A delegated session keeps its Atom parent.
    attributes.update(
        {observability.ATOM_SESSION_ID_ATTR: "child", observability.ATOM_SESSION_PARENT_ATTR: "root-atom"}
    )
    assert observability._current_atom_session() == ("child", "root-atom")


def test_a_session_opened_inside_a_code_cell_records_that_cell_as_its_spawner(monkeypatch) -> None:
    from responses_api_agents.nooa_agent import observability

    sessions = iter([("root-session", None), ("child-session", "root-session"), ("child-session", "root-session")])
    monkeypatch.setattr(observability, "_current_atom_session", lambda: next(sessions))
    trace = GymTraceHooks()
    root = trace.before_agent_call(call_id="call-root", parent_call_id=None)
    cell = trace.before_code_execution(code="await self.delegate('x', 'y')", execution_id="cell-1")
    child = trace.before_agent_call(call_id="call-child", parent_call_id=None)
    trace.after_agent_call(context=child, exception=None)
    # A later turn of the same session was not spawned by anything.
    later = trace.before_agent_call(call_id="call-child-2", parent_call_id="call-child")
    trace.after_agent_call(context=later, exception=None)
    trace.after_code_execution(context=cell, result=None, exception=None)
    trace.after_agent_call(context=root, exception=None)

    invocations = trace._invocations
    assert invocations["call-child"].spawned_by_tool_call_id == "cell-1"
    assert invocations["call-child"].parent_invocation_id == "call-root"
    assert invocations["call-child-2"].spawned_by_tool_call_id is None
    assert invocations["call-root"].spawned_by_tool_call_id is None
    assert trace._current_tool.get() is None
