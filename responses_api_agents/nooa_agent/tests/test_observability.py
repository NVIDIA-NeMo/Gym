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

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ToolCallObservation
from responses_api_agents.nooa_agent.gym_llm import GymModelCall, RolloutLLMState
from responses_api_agents.nooa_agent.observability import (
    GymTraceHooks,
    ensure_verifier_final_message,
    finalize_observation_gaps,
)


def response(
    *output: object, response_id: str = "resp-1", usage: NeMoGymResponseUsage | None = None
) -> NeMoGymResponse:
    return NeMoGymResponse(
        id=response_id,
        created_at=0,
        model="policy",
        object="response",
        output=list(output),
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=usage,
    )


def test_projects_native_model_and_tool_evidence() -> None:
    trace = GymTraceHooks()
    state = RolloutLLMState(max_policy_calls=1)
    context = trace.before_agent_call(call_id="root", parent_call_id=None)
    params = NeMoGymResponseCreateParamsNonStreaming(input="calculate")
    call = GymModelCall(
        model_ref=ModelServerRef(type="responses_api_models", name="policy"),
        request=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "calculate"}]),
        response=response(
            NeMoGymResponseFunctionToolCall(
                id="fc-1",
                call_id="call-1",
                name="execute_python",
                arguments='{"code":"print(7)"}',
            )
        ),
    )
    trace.on_model_call(call)
    state.calls.append(call)
    execution = trace.before_code_execution(code="print(7)", execution_id="exec-1", tool_call_id="call-1")
    trace.after_code_execution(context=execution, result={"stdout": "7"}, exception=None)
    trace.after_agent_call(context=context, exception=None)

    episode = trace.project(create_params=params, state=state)

    invocation = next(record for record in episode.observations.records if isinstance(record, AgentInvocation))
    tool = next(record for record in episode.observations.records if isinstance(record, ToolCallObservation))
    assert invocation.invocation_id == "root"
    assert invocation.model_calls[0].response_id == "resp-1"
    assert [item.type for item in invocation.conversation] == ["message", "function_call", "function_call_output"]
    assert tool.tool_call_id == "call-1"
    assert tool.status == "completed"
    assert json.loads(episode.response.output[-1].output) == {"stdout": "7"}


def test_no_model_call_preserves_original_responses_input() -> None:
    trace = GymTraceHooks()
    params = NeMoGymResponseCreateParamsNonStreaming(input="original row input")

    episode = trace.project(create_params=params, state=RolloutLLMState(max_policy_calls=1))

    invocation = next(record for record in episode.observations.records if isinstance(record, AgentInvocation))
    assert invocation.invocation_id == "root"
    assert invocation.conversation[0].content == "original row input"
    assert episode.response.output == []


@pytest.mark.parametrize("error", [ValueError("bad code"), asyncio.CancelledError()])
def test_failed_and_cancelled_tools_keep_matching_output(error: BaseException) -> None:
    trace = GymTraceHooks()
    context = trace.before_agent_call(call_id="root", parent_call_id=None)
    execution = trace.before_code_execution(code="raise ValueError()", execution_id="exec-1", tool_call_id="call-1")
    trace.after_code_execution(context=execution, result=None, exception=error)
    trace.after_agent_call(context=context, exception=error)

    episode = trace.project(
        create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
        state=RolloutLLMState(max_policy_calls=1),
    )

    invocation = next(record for record in episode.observations.records if isinstance(record, AgentInvocation))
    tool = next(record for record in episode.observations.records if isinstance(record, ToolCallObservation))
    assert tool.status == ("cancelled" if isinstance(error, asyncio.CancelledError) else "failed")
    assert invocation.status == "failed"
    assert json.loads(episode.response.output[-1].output)["error_type"] == type(error).__name__


@pytest.mark.parametrize("missing_usage", [False, True])
def test_episode_usage_sums_only_complete_usage(missing_usage: bool) -> None:
    trace = GymTraceHooks()
    state = RolloutLLMState(max_policy_calls=2)
    params = NeMoGymResponseCreateParamsNonStreaming(input=[])
    for index in range(2):
        usage = None
        if not (missing_usage and index == 1):
            usage = NeMoGymResponseUsage.model_validate(
                {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                }
            )
        call = GymModelCall(
            model_ref=ModelServerRef(type="responses_api_models", name=f"model-{index}"),
            request=params,
            response=response(
                NeMoGymResponseOutputMessage(
                    id=f"message-{index}",
                    content=[NeMoGymResponseOutputText(annotations=[], text=str(index))],
                ),
                response_id=f"response-{index}",
                usage=usage,
            ),
        )
        trace.on_model_call(call)
        state.calls.append(call)

    episode = trace.project(create_params=params, state=state)

    if missing_usage:
        assert episode.response.usage is None
    else:
        assert episode.response.usage is not None
        assert episode.response.usage.total_tokens == 24


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
    assert [gap.code for gap in gaps] == ["non_trainable_fallback_output"]
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
    assert [gap.code for gap in gaps] == ["non_trainable_fallback_output"]


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


def test_finalize_observation_gaps_appends_termination_gap() -> None:
    bundle = AgentObservationBundle(source="nooa", records=[], gaps=[])

    finalized = finalize_observation_gaps(
        bundle,
        termination_reason="policy_budget_exceeded",
        termination_error="budget exhausted",
    )

    assert finalized.gaps[0].code == "policy_budget_exceeded"
    assert finalized.gaps[0].detail == "budget exhausted"
