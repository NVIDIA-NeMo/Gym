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

import pytest

from nemo_gym.base_responses_api_model import ModelCallRecord
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.rollout_observability import (
    AgentInvocation,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
    join_model_call_observations,
)
from responses_api_agents.nooa_agent.gym_tools import GymToolExecution
from responses_api_agents.nooa_agent.observability import (
    GymTraceHooks,
    NOOATraceSnapshot,
    project_nooa_result,
)


def response(response_id: str, text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id=response_id,
        created_at=0,
        model="policy",
        object="response",
        output=[
            NeMoGymResponseOutputMessageForTraining(
                id=f"message-{response_id}",
                content=[NeMoGymResponseOutputText(annotations=[], text=text)],
                prompt_token_ids=[1, 2],
                generation_token_ids=[3],
                generation_log_probs=[-0.1],
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def test_projects_model_and_semantic_resource_evidence_in_execution_order() -> None:
    first = response("resp-1", "I will check.")
    final = response("resp-2", "It is cold.")
    execution = GymToolExecution(
        tool_call_id="tool-1",
        name="get_weather",
        arguments={"city": "Paris"},
        output={"weather": "cold"},
        status="completed",
        started_at=1.0,
        completed_at=1.1,
        duration_ms=100,
        invocation_id="child-call",
    )
    params = NeMoGymResponseCreateParamsNonStreaming(input="Weather in Paris?")

    hooks = GymTraceHooks(ModelServerRef(type="responses_api_models", name="policy_model"))
    root_context = hooks.before_agent_call(None, "answer", (), {}, "root-call", None)
    hooks.record_model_response(first)
    child_context = hooks.before_agent_call(None, "helper", (), {}, "child-call", "root-call")
    hooks.record_tool_execution(execution)
    hooks.after_agent_call(None, "helper", None, None, child_context)
    hooks.record_model_response(final)
    hooks.after_agent_call(None, "answer", "done", None, root_context)

    projected, bundle = project_nooa_result(
        responses_create_params=params,
        return_value="ignored fallback",
        model_responses=[first, final],
        tool_executions=[execution],
        trace=hooks.snapshot(),
    )

    assert [item.type for item in projected.output] == [
        "message",
        "function_call",
        "function_call_output",
        "message",
        "message",
    ]
    assert projected.output[0].generation_token_ids == [3]
    assert projected.output[-2].generation_log_probs == [-0.1]
    assert projected.output[-1].content[0].text == "ignored fallback"
    invocations = [record for record in bundle.records if isinstance(record, AgentInvocation)]
    assert invocations[1].parent_invocation_id == "root-call"
    assert [ref.response_id for ref in invocations[0].model_calls] == ["resp-1", "resp-2"]
    tool = next(record for record in bundle.records if isinstance(record, ToolCallObservation))
    assert tool.invocation_id == "child-call"
    assert tool.duration_ms == 100
    assert [gap.code for gap in bundle.gaps] == ["non_trainable_terminal_output"]

    joined = join_model_call_observations(
        bundle,
        [
            ModelCallRecord(
                call_index=index,
                model_call_id=f"captured-{index}",
                response_id=response_id,
                model_ref=ModelServerRef(type="responses_api_models", name="policy_model"),
            )
            for index, response_id in enumerate(("resp-1", "resp-2"))
        ],
    )
    joined_root = next(
        record
        for record in joined.records
        if isinstance(record, AgentInvocation) and record.invocation_id == "root-call"
    )
    assert [reference.model_call_id for reference in joined_root.model_calls] == ["captured-0", "captured-1"]
    assert [gap.code for gap in joined.gaps] == ["non_trainable_terminal_output"]


def test_marks_return_value_fallback_as_non_trainable() -> None:
    params = NeMoGymResponseCreateParamsNonStreaming(input="Hello")

    projected, bundle = project_nooa_result(
        responses_create_params=params,
        return_value={"answer": "fallback"},
        model_responses=[],
        tool_executions=[],
        trace=NOOATraceSnapshot(),
    )

    assert projected.output[0].content[0].text == '{"answer": "fallback"}'
    assert [gap.code for gap in bundle.gaps] == ["non_trainable_terminal_output"]


def test_preserves_llm_observation_gaps() -> None:
    gap = ObservationGap(
        code="prior_output_metadata_unrestored",
        detail="Prior generated tokens may be masked.",
    )

    _, bundle = project_nooa_result(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="Hello"),
        return_value=None,
        model_responses=[],
        tool_executions=[],
        trace=NOOATraceSnapshot(),
        result_present=False,
        observation_gaps=[gap],
    )

    assert bundle.gaps == [gap]


def test_projected_model_output_is_consumable_by_training_converter() -> None:
    authored = response("resp-training", "train me")
    params = NeMoGymResponseCreateParamsNonStreaming(input="Question")
    projected, _ = project_nooa_result(
        responses_create_params=params,
        return_value="ignored",
        model_responses=[authored],
        tool_executions=[],
        trace=NOOATraceSnapshot(
            output=list(authored.output),
            model_calls={
                "root": [
                    ModelCallRef(
                        model_ref=ModelServerRef(type="responses_api_models", name="policy_model"),
                        response_id="resp-training",
                    )
                ]
            },
        ),
    )

    converted = ResponsesConverter(return_token_id_information=True).responses_to_chat_completion_create_params(
        NeMoGymResponseCreateParamsNonStreaming.model_validate(
            {"input": [item.model_dump(mode="json") for item in projected.output]}
        )
    )

    assert converted.messages[0]["prompt_token_ids"] == [1, 2]
    assert converted.messages[0]["generation_token_ids"] == [3]
    assert converted.messages[0]["generation_log_probs"] == [-0.1]


@pytest.mark.asyncio
async def test_concurrent_sibling_invocations_keep_model_attribution_isolated() -> None:
    hooks = GymTraceHooks(ModelServerRef(type="responses_api_models", name="policy_model"))
    ready = asyncio.Event()
    entered = 0

    async def observe(call_id: str, response_id: str) -> None:
        nonlocal entered
        context = hooks.before_agent_call(None, "child", (), {}, call_id, "root-call")
        entered += 1
        if entered == 2:
            ready.set()
        await ready.wait()
        hooks.record_model_response(response(response_id, response_id))
        await asyncio.sleep(0)
        hooks.after_agent_call(None, "child", response_id, None, context)

    await asyncio.gather(observe("child-a", "response-a"), observe("child-b", "response-b"))
    snapshot = hooks.snapshot()

    assert {
        invocation.invocation_id: [reference.response_id for reference in invocation.model_calls]
        for invocation in snapshot.invocations
    } == {"child-a": ["response-a"], "child-b": ["response-b"]}


def test_typed_return_value_overrides_nonterminal_model_message() -> None:
    intermediate = response("resp-intermediate", "working")
    hooks = GymTraceHooks(ModelServerRef(type="responses_api_models", name="policy_model"))
    hooks.record_model_response(intermediate)

    projected, bundle = project_nooa_result(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="Question"),
        return_value={"answer": 42},
        model_responses=[intermediate],
        tool_executions=[],
        trace=hooks.snapshot(),
    )

    messages = [item for item in projected.output if item.type == "message"]
    assert [message.content[0].text for message in messages] == ["working", '{"answer": 42}']
    assert [gap.code for gap in bundle.gaps] == ["non_trainable_terminal_output"]


def test_matching_earlier_message_does_not_suppress_terminal_result_after_tool() -> None:
    authored = response("resp-equal", "done")
    hooks = GymTraceHooks(ModelServerRef(type="responses_api_models", name="policy_model"))
    hooks.record_model_response(authored)
    hooks.record_tool_execution(
        GymToolExecution(
            tool_call_id="tool-after-message",
            name="lookup",
            arguments={},
            output="side effect",
            status="completed",
            started_at=1.0,
            completed_at=1.1,
            duration_ms=100.0,
        )
    )

    projected, bundle = project_nooa_result(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="Question"),
        return_value="done",
        model_responses=[authored],
        tool_executions=[],
        trace=hooks.snapshot(),
    )

    assert [item.type for item in projected.output] == [
        "message",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert projected.output[-1].content[0].text == "done"
    assert [gap.code for gap in bundle.gaps] == ["non_trainable_terminal_output"]


def test_successful_none_result_projects_null_but_termination_without_result_does_not() -> None:
    params = NeMoGymResponseCreateParamsNonStreaming(input="Question")

    completed, completed_bundle = project_nooa_result(
        responses_create_params=params,
        return_value=None,
        model_responses=[],
        tool_executions=[],
        trace=NOOATraceSnapshot(),
        result_present=True,
    )
    terminated, terminated_bundle = project_nooa_result(
        responses_create_params=params,
        return_value=None,
        model_responses=[],
        tool_executions=[],
        trace=NOOATraceSnapshot(),
        result_present=False,
        termination_reason="policy_budget_exceeded",
    )

    assert completed.output[-1].content[0].text == "null"
    assert [gap.code for gap in completed_bundle.gaps] == ["non_trainable_terminal_output"]
    assert terminated.output == []
    assert [gap.code for gap in terminated_bundle.gaps] == ["policy_budget_exceeded"]
