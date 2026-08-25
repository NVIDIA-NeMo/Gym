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

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

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
    ToolCallObservation,
    join_model_call_observations,
)
from responses_api_agents.nooa_agent.gym_tools import GymToolExecution
from responses_api_agents.nooa_agent.observability import TraceEvent, project_nooa_result


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


def events() -> list[SimpleNamespace]:
    started = datetime.now(UTC)
    return [
        SimpleNamespace(
            event_type="BeforeAgentCall",
            call_id="root-call",
            parent_call_id=None,
            is_top_level=True,
            timestamp=started,
        ),
        SimpleNamespace(
            event_type="BeforeAgentCall",
            call_id="child-call",
            parent_call_id="root-call",
            is_top_level=False,
            timestamp=started + timedelta(milliseconds=1),
        ),
        SimpleNamespace(
            event_type="AfterAgentCall",
            call_id="child-call",
            success=True,
            exception_type=None,
            timestamp=started + timedelta(milliseconds=3),
        ),
        SimpleNamespace(
            event_type="AfterAgentCall",
            call_id="root-call",
            success=True,
            exception_type=None,
            timestamp=started + timedelta(milliseconds=5),
        ),
    ]


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

    projected, bundle = project_nooa_result(
        responses_create_params=params,
        return_value="ignored fallback",
        model_responses=[first, final],
        tool_executions=[execution],
        timeline=[
            TraceEvent(kind="model", value=first, invocation_id="root-call"),
            TraceEvent(kind="tool", value=execution, invocation_id="child-call"),
            TraceEvent(kind="model", value=final, invocation_id="root-call"),
        ],
        nooa_events=events(),
        model_ref=ModelServerRef(type="responses_api_models", name="policy_model"),
    )

    assert [item.type for item in projected.output] == [
        "message",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert projected.output[0].generation_token_ids == [3]
    assert projected.output[-1].generation_log_probs == [-0.1]
    invocations = [record for record in bundle.records if isinstance(record, AgentInvocation)]
    assert invocations[1].parent_invocation_id == "root-call"
    assert [ref.response_id for ref in invocations[0].model_calls] == ["resp-1", "resp-2"]
    tool = next(record for record in bundle.records if isinstance(record, ToolCallObservation))
    assert tool.invocation_id == "child-call"
    assert tool.duration_ms == 100
    assert bundle.gaps == []

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
    assert joined.gaps == []


def test_marks_return_value_fallback_as_non_trainable() -> None:
    params = NeMoGymResponseCreateParamsNonStreaming(input="Hello")

    projected, bundle = project_nooa_result(
        responses_create_params=params,
        return_value={"answer": "fallback"},
        model_responses=[],
        tool_executions=[],
        timeline=[],
        nooa_events=[],
        model_ref=ModelServerRef(type="responses_api_models", name="policy_model"),
    )

    assert projected.output[0].content[0].text == '{"answer": "fallback"}'
    assert [gap.code for gap in bundle.gaps] == ["non_trainable_fallback_output"]


def test_projected_model_output_is_consumable_by_training_converter() -> None:
    authored = response("resp-training", "train me")
    params = NeMoGymResponseCreateParamsNonStreaming(input="Question")
    projected, _ = project_nooa_result(
        responses_create_params=params,
        return_value="ignored",
        model_responses=[authored],
        tool_executions=[],
        timeline=[TraceEvent(kind="model", value=authored, invocation_id="root")],
        nooa_events=[],
        model_ref=ModelServerRef(type="responses_api_models", name="policy_model"),
    )

    converted = ResponsesConverter(return_token_id_information=True).responses_to_chat_completion_create_params(
        NeMoGymResponseCreateParamsNonStreaming.model_validate(
            {"input": [item.model_dump(mode="json") for item in projected.output]}
        )
    )

    assert converted.messages[0]["prompt_token_ids"] == [1, 2]
    assert converted.messages[0]["generation_token_ids"] == [3]
    assert converted.messages[0]["generation_log_probs"] == [-0.1]
