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

import pytest
from pydantic import BaseModel

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_collection import _build_trajectory_record
from nemo_gym.rollout_observability import AgentObservationBundle, TrajectoryRecord
from responses_api_agents.nooa_agent.gym_llm import GymModelCall, RolloutLLMState
from responses_api_agents.nooa_agent.observability import (
    GymTraceHooks,
    _json_output,
    ensure_verifier_final_message,
    finalize_observation_gaps,
)
from responses_api_agents.nooa_agent.tests.test_gym_llm import model_response


def test_scoped_projection_preserves_model_ownership_and_collector_tool_output_join() -> None:
    trace = GymTraceHooks()
    state = RolloutLLMState(max_steps=1)
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


@pytest.mark.parametrize("error", [ValueError("bad code"), asyncio.CancelledError()])
def test_failed_and_cancelled_code_keeps_matching_output(error: BaseException) -> None:
    trace = GymTraceHooks()
    context = trace.before_agent_call(call_id="root", parent_call_id=None)
    execution = trace.before_code_execution(code="raise ValueError()", execution_id="exec-1", tool_call_id="call-1")
    trace.after_code_execution(context=execution, result=None, exception=error)
    trace.after_agent_call(context=context, exception=error)
    episode, trajectory = trace.project(
        create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
        state=RolloutLLMState(max_steps=1),
        task_id="task",
        rollout_id="0-0",
    )
    assert trajectory.tool_calls[0].status == ("cancelled" if isinstance(error, asyncio.CancelledError) else "failed")
    assert trajectory.invocations[0].status == "failed"
    output = episode.response.output[-1]
    assert output.call_id == trajectory.tool_calls[0].tool_call_id
    assert json.loads(output.output)["error_type"] == type(error).__name__


class TypedAnswer(BaseModel):
    value: int


@pytest.mark.parametrize("missing_usage", [False, True])
def test_episode_usage_sums_all_models_only_when_complete(missing_usage: bool) -> None:
    trace = GymTraceHooks()
    state = RolloutLLMState(max_steps=2)
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
    episode, _ = trace.project(create_params=params, state=state, task_id="t", rollout_id="r")
    if missing_usage:
        assert episode.response.usage is None
    else:
        usage = episode.response.usage
        assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (20, 4, 24)
        assert usage.input_tokens_details.cached_tokens == 6
        assert usage.output_tokens_details.reasoning_tokens == 2


def test_failed_resource_and_pending_model_keep_evidence() -> None:
    trace = GymTraceHooks()
    state = RolloutLLMState(max_steps=1)
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
        termination_reason="policy_budget_exceeded",
        termination_error="budget exhausted",
    )

    assert finalized.gaps[0].code == "policy_budget_exceeded"
    assert finalized.gaps[0].detail == "budget exhausted"
