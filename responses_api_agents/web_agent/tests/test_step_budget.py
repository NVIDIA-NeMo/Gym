# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from nemo_gym.web import WebTask
from responses_api_agents.web_agent.app import WebAgentRunRequest
from responses_api_agents.web_agent.tests.test_app import _agent, _native_model_response, _observation, _seed, _wire


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("steps", [1, 2, 3, 5])
@pytest.mark.asyncio
async def test_step_budget_preserves_history_and_changes_only_terminal_template_options(enabled, steps):
    agent = _agent(parse_retries=0, max_steps=steps, nano_omni_stop_at_step_budget=enabled)
    calls = _wire(
        agent,
        {
            "/seed_session": [_seed()],
            "/v1/responses": [
                _native_model_response("computer", '{"actions":[{"action":"wait","duration":1}]}')
                for _ in range(steps)
            ],
            "/step": [
                {"operation_id": f"step-{i}", "observation": _observation(), "execution_ok": True, "terminated": False}
                for i in range(steps)
            ],
            "/evaluate": [{"result": {"reward": 0, "valid_sample": True}}],
            "/close": [{"closed": True}],
        },
    )
    original = {"truncate_history_thinking": False, "enable_thinking": True, "unrelated": "retained"}
    result = await agent.run(
        SimpleNamespace(cookies={}),
        WebAgentRunRequest(
            responses_create_params={
                "input": [],
                "metadata": {"chat_template_kwargs": json.dumps(original), "tag": "retained"},
            },
            web_task=WebTask(benchmark="visualwebarena", task_id="0", intent="Observe the step budget"),
        ),
    )
    requests = [value for _, path, value in calls if path == "/v1/responses"]
    assert len(requests) == steps and result.truncated and not result.terminated
    for index, request in enumerate(requests):
        expected = {**original, "enable_thinking": False} if enabled and index + 2 >= steps else original
        assert json.loads(request.metadata["chat_template_kwargs"]) == expected
        assert request.metadata["tag"] == "retained"
        # The warning is an observation in the retained history, not rewritten
        # model output or a fabricated terminal action.
        warning = "End the interaction now with only a terminate tool call"
        assert (warning in request.model_dump_json()) is (enabled and steps >= 2 and index + 2 >= steps)
        if warning in request.model_dump_json():
            assert 'answer=""' in str(request.input)
            assert 'answer="N/A"' not in str(request.input)
    assert next(value for _, path, value in calls if path == "/evaluate")["final_answer"] is None


@pytest.mark.asyncio
async def test_requested_terminal_answer_is_evaluated_before_session_close():
    agent = _agent(parse_retries=0, max_steps=3, nano_omni_stop_at_step_budget=True)
    calls = _wire(
        agent,
        {
            "/seed_session": [_seed()],
            "/v1/responses": [
                _native_model_response("computer", '{"actions":[{"action":"wait","duration":1}]}'),
                _native_model_response("terminate", '{"status":"failure","answer":""}'),
            ],
            "/step": [
                {"operation_id": "step-0", "observation": _observation(), "execution_ok": True, "terminated": False},
                {"operation_id": "step-1", "observation": _observation(), "execution_ok": True, "terminated": True},
            ],
            "/evaluate": [{"result": {"reward": 0, "valid_sample": True}}],
            "/close": [{"closed": True}],
        },
    )
    result = await agent.run(
        SimpleNamespace(cookies={}),
        WebAgentRunRequest(
            responses_create_params={"input": []},
            web_task=WebTask(benchmark="visualwebarena", task_id="0"),
        ),
    )
    assert result.terminated and not result.truncated and not result.mask_sample
    assert next(body for _, path, body in calls if path == "/evaluate")["final_answer"] == ""
    assert [path for _, path, _ in calls][-2:] == ["/evaluate", "/close"]
