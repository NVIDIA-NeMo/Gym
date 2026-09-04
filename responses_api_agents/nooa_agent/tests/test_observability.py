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

from nooa.atif import AgentSchema, ObservationResultSchema, ObservationSchema, StepObject, ToolCallSchema, Trajectory

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ModelCallRef, ToolCallObservation
from responses_api_agents.nooa_agent.observability import (
    adapt_response_for_verify,
    finalize_observations,
    project_nooa_episode,
)


def test_projects_atif_semantics_and_model_references() -> None:
    trajectory = Trajectory(
        trajectory_id="root",
        agent=AgentSchema(name="TestAgent", version="1"),
        steps=[
            StepObject(step_id=1, source="system", message="Use tools."),
            StepObject(step_id=2, source="user", message="Weather in Paris?"),
            StepObject(
                step_id=3,
                timestamp="2026-09-02T12:00:00.000Z",
                source="agent",
                message="I will check.",
                reasoning_content="Need current weather.",
                tool_calls=[
                    ToolCallSchema(
                        tool_call_id="call-1",
                        function_name="execute_python",
                        arguments={"code": "self.get_weather(city='Paris')"},
                    )
                ],
                observation=ObservationSchema(
                    results=[ObservationResultSchema(source_call_id="call-1", content="returned_value:\ncold")]
                ),
                llm_call_count=1,
            ),
            StepObject(step_id=4, source="agent", message="It is cold.", llm_call_count=1),
        ],
    )
    model_calls = [
        ModelCallRef(
            model_ref=ModelServerRef(type="responses_api_models", name="policy"),
            response_id="response-1",
        ),
        ModelCallRef(
            model_ref=ModelServerRef(type="responses_api_models", name="policy"),
            response_id="response-2",
        ),
    ]

    episode = project_nooa_episode(
        create_params=NeMoGymResponseCreateParamsNonStreaming(input="Weather in Paris?"),
        trajectory=trajectory,
        model_calls=model_calls,
    )

    assert [item.type for item in episode.response.output] == [
        "reasoning",
        "message",
        "function_call",
        "function_call_output",
        "message",
    ]
    invocation = next(record for record in episode.observations.records if isinstance(record, AgentInvocation))
    assert [reference.response_id for reference in invocation.model_calls] == ["response-1", "response-2"]
    assert [item.type for item in invocation.conversation] == [
        "message",
        "message",
        "reasoning",
        "message",
        "function_call",
        "function_call_output",
        "message",
    ]
    tool = next(record for record in episode.observations.records if isinstance(record, ToolCallObservation))
    assert tool.tool_name == "execute_python"
    assert tool.status == "completed"
    assert tool.timing_source == "artifact"
    assert episode.observations.gaps == []


def test_preserves_model_authored_return_result_without_synthetic_message() -> None:
    trajectory = Trajectory(
        trajectory_id="root",
        agent=AgentSchema(name="TestAgent", version="1"),
        steps=[
            StepObject(
                step_id=1,
                source="agent",
                message="",
                tool_calls=[
                    ToolCallSchema(
                        tool_call_id="return-1",
                        function_name="return_result",
                        arguments={"result": "fallback"},
                    )
                ],
                llm_call_count=1,
            )
        ],
    )

    episode = project_nooa_episode(
        create_params=NeMoGymResponseCreateParamsNonStreaming(input="Hello"),
        trajectory=trajectory,
        model_calls=[],
    )

    assert [item.type for item in episode.response.output] == ["function_call"]
    assert episode.response.output[0].name == "return_result"
    assert episode.observations.gaps == []
    invocation = next(record for record in episode.observations.records if isinstance(record, AgentInvocation))
    assert [item.type for item in invocation.conversation] == ["function_call"]


def test_reports_ambiguous_model_ownership_for_nested_trajectories() -> None:
    child = Trajectory(
        trajectory_id="child",
        agent=AgentSchema(name="ChildAgent", version="1"),
        steps=[StepObject(step_id=1, source="agent", message="child answer", llm_call_count=1)],
    )
    root = Trajectory(
        trajectory_id="root",
        agent=AgentSchema(name="RootAgent", version="1"),
        steps=[StepObject(step_id=1, source="agent", message="root answer", llm_call_count=1)],
        subagent_trajectories=[child],
    )
    model_call = ModelCallRef(
        model_ref=ModelServerRef(type="responses_api_models", name="policy"),
        response_id="response-1",
    )

    episode = project_nooa_episode(
        create_params=NeMoGymResponseCreateParamsNonStreaming(input="Delegate"),
        trajectory=root,
        model_calls=[model_call],
    )

    invocations = [record for record in episode.observations.records if isinstance(record, AgentInvocation)]
    assert invocations[1].parent_invocation_id == "root"
    assert all(not invocation.model_calls for invocation in invocations)
    assert [gap.code for gap in episode.observations.gaps] == ["model_call_ownership_unavailable"]


def test_adapt_response_for_verify_adds_fallback_without_mutating_episode() -> None:
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

    adapted, gaps = adapt_response_for_verify(response, "fallback answer")

    assert adapted.output[0].content[0].text == "fallback answer"
    assert [gap.code for gap in gaps] == ["non_trainable_fallback_output"]
    assert response.output == []


def test_finalize_observations_appends_termination_gap() -> None:
    bundle = AgentObservationBundle(source="nooa", records=[], gaps=[])

    finalized = finalize_observations(
        bundle,
        termination_reason="policy_budget_exceeded",
        termination_error="budget exhausted",
    )

    assert finalized.gaps[0].code == "policy_budget_exceeded"
    assert finalized.gaps[0].detail == "budget exhausted"
