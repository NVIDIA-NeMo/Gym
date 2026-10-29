# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.nooa_agent.gym_llm import GymModelCall, RolloutLLMState
from responses_api_agents.nooa_agent.observability import GymTraceHooks
from responses_api_agents.nooa_agent.result import finalize_run_result
from responses_api_agents.nooa_agent.runner import NOOARunResult
from tests.unit_tests.test_terminal_attribution import _assemble, _assistant_item, _entry


@pytest.mark.parametrize("return_value,expected_call", [("42", "child"), ("computed result: 42", "root")])
def test_root_return_preserves_served_child_or_declares_synthetic_terminal(return_value, expected_call):
    trace = GymTraceHooks()
    state = RolloutLLMState(max_policy_calls=None)
    trace.before_agent_call(call_id="root", parent_call_id=None)
    entries = []
    for name, text in [("root", "delegate"), ("child", "42")]:
        if name == "child":
            trace.before_agent_call(call_id="child", parent_call_id="root")
        call = GymModelCall(
            model_ref=ModelServerRef(type="responses_api_models", name="policy"),
            request=NeMoGymResponseCreateParamsNonStreaming(input="task"),
        )
        trace.on_model_call(call)
        state.calls.append(call)
        item = _assistant_item(text, item_id=f"{name}-message")
        call.response = NeMoGymResponse(
            id=f"resp_{name}",
            model="m",
            created_at=1,
            object="response",
            output=[item],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        entry = _entry(name, [1, 2] if name == "root" else [8, 9], [3, 4], text=text, response_id=f"resp_{name}")
        entry.output_items = [item]
        entries.append(entry)
    episode, _ = trace.project(
        create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
        state=state,
        task_id="t",
        rollout_id="r",
    )
    response, observations = finalize_run_result(
        NOOARunResult(
            episode=episode,
            return_value=return_value,
            model_cookies={},
            resource_cookies={},
        )
    )
    declaration = (response.metadata or {}).get("terminal_response_id")
    assert declaration == ("resp_root" if expected_call == "root" else None)
    built = _assemble(
        "r",
        entries,
        "prefix_merging",
        "m",
        verified_response=response.model_dump(mode="json"),
        declared_response_id=declaration,
    )
    assert built["mask_sample"] is False
    assert built["metrics"]["terminal_attribution"]["call_id"] == expected_call
    assert any(gap.code == "non_trainable_terminal_output" for gap in observations.gaps) == (expected_call == "root")
    assert episode.response.metadata is None
