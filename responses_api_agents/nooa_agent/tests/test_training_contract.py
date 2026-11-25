# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Root-call identity and verifier-only text through the actual episode result."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServer
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseOutputMessage
from nemo_gym.server_runtime import SingleAgentTurnResult
from responses_api_agents.nooa_agent.gym_llm import GymModelCall, RolloutLLMState
from responses_api_agents.nooa_agent.observability import GymTraceHooks, ensure_verifier_final_message
from responses_api_agents.nooa_agent.tests.test_environment import _request, _Response
from responses_api_agents.nooa_agent.tests.test_gym_llm import model_response


def projected_response(*, later_root_turn=False):
    trace = GymTraceHooks()
    state = RolloutLLMState(max_policy_calls=None)
    params = NeMoGymResponseCreateParamsNonStreaming(input="task")

    def capture(identity, *, auxiliary=False):
        call = GymModelCall(
            model_ref=ModelServerRef(type="responses_api_models", name="policy"),
            request=params,
            response=NeMoGymResponse.model_validate(
                model_response(
                    NeMoGymResponseOutputMessage(
                        id=identity + "-item", content=[{"type": "output_text", "text": identity, "annotations": []}]
                    ),
                    response_id=identity,
                )
            ),
            auxiliary=auxiliary,
        )
        trace.on_model_call(call)
        state.calls.append(call)

    wrapper = trace.before_agent_call(call_id="wrapper", parent_call_id=None)
    agent = SimpleNamespace(agent_id="stable-root-agent")
    root = trace.before_agent_call(call_id="root-agent", parent_call_id="wrapper", agent=agent)
    capture("root-response")
    trace.after_agent_call(context=root, exception=None)
    if later_root_turn:
        root = trace.before_agent_call(call_id="root-turn-2", parent_call_id="wrapper", agent=agent)
        capture("root-response-2")
        trace.after_agent_call(context=root, exception=None)
    summary = trace.before_agent_call(call_id="summary", parent_call_id="wrapper")
    capture("late-summary", auxiliary=True)
    trace.after_agent_call(context=summary, exception=None)
    child = trace.before_agent_call(call_id="child-agent", parent_call_id="root-agent")
    capture("late-child")
    trace.after_agent_call(context=child, exception=None)
    trace.after_agent_call(context=wrapper, exception=None)
    episode, _ = trace.project(
        create_params=params, state=state, task_id="task", rollout_id="rollout", default_model="policy"
    )
    return episode.response


def test_later_root_turn_uses_stable_agent_identity_before_late_auxiliary_and_child_calls():
    response = projected_response(later_root_turn=True)
    assert response.id == "root-response-2"
    assert response.output[-1].id == "late-child-item"


async def test_verifier_gets_synthetic_answer_and_result_keeps_only_captured_actions():
    captured = projected_response()
    assert captured.id == "root-response"
    verifier_response, _ = ensure_verifier_final_message(captured, {"answer": "structured result"})
    assert verifier_response.output[-1].synthetic
    assert len(captured.output) == 3
    received = []

    async def verify(**kwargs):
        body = kwargs["json"]
        received.append(body["response"])
        return _Response(body | {"reward": 1.0})

    environment = SimpleNamespace(
        server_client=SimpleNamespace(post=AsyncMock(side_effect=verify)),
        config=SimpleNamespace(resources_server=SimpleNamespace(name="resources"), forward_agent_evidence=False),
        _verify_body=SingleAgentTurnEnvironmentServer._verify_body,
    )
    result = await SingleAgentTurnEnvironmentServer._verify(
        environment, _request(), SimpleNamespace(resources_cookies={}), verifier_response, None
    )
    assert received[0]["output"][-1]["synthetic"] is True
    assert "structured result" in received[0]["output"][-1]["content"][0]["text"]
    assert result.reward == 1.0 and result.response.id == "root-response"
    assert result.response.output == captured.output
    assert result.verifier_response == verifier_response
    assert SingleAgentTurnResult.model_validate_json(result.model_dump_json()) == result


def test_ordinary_output_and_unmarked_edits_are_never_dropped():
    response = projected_response()
    ordinary = SingleAgentTurnResult(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"), response=response, reward=0
    )
    assert ordinary.verifier_response is None and ordinary.response == response
    assert "verifier_response" not in ordinary.model_dump()
    assert all("synthetic" not in item.model_dump() for item in ordinary.response.output)
