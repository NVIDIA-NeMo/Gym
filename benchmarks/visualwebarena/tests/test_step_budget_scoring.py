# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nemo_gym.web import WebTask
from resources_servers.webarena_browser.reference_evaluation import classic_evaluation, visualwebarena_evaluation
from responses_api_agents.web_agent.app import WebAgentRunRequest
from responses_api_agents.web_agent.tests.test_app import (
    _agent,
    _FakeHttpResponse,
    _native_model_response,
    _observation,
    _seed,
)


@pytest.mark.parametrize("steps,expected_answer,expected_reward", [(2, "", 0), (5, "N/A", 1)])
@pytest.mark.asyncio
async def test_budget_stop_cannot_claim_the_unachievable_reference(
    monkeypatch, steps, expected_answer, expected_reward
):
    agent = _agent(parse_retries=0, max_steps=steps, nano_omni_stop_at_step_budget=True)
    judge = Mock(side_effect=AssertionError("Neither an empty answer nor exact N/A needs a judge"))
    monkeypatch.setattr(classic_evaluation, "_judge_chat", judge)
    evaluated = []

    async def post(server_name, url_path, **kwargs):
        body = kwargs["json"]
        if url_path == "/seed_session":
            result = _seed()
        elif url_path == "/v1/responses":
            # Model boundary only: obey the actual rendered budget instruction.
            # Without that instruction the policy independently reports N/A.
            instruction = re.search(r'status="([^"]+)", answer="([^"]*)"', str(body.input))
            status, answer = instruction.groups() if instruction else ("success", "N/A")
            result = _native_model_response("terminate", json.dumps({"status": status, "answer": answer}))
        elif url_path == "/step":
            result = {
                "operation_id": "budget-step",
                "observation": _observation(),
                "execution_ok": True,
                "terminated": True,
            }
        elif url_path == "/evaluate":
            evaluated.append(body["final_answer"])
            reward = visualwebarena_evaluation._string_match(
                {"intent": "Find an unavailable item", "eval": {"reference_answers": {"fuzzy_match": "N/A"}}},
                body["final_answer"],
            )
            result = {"result": {"reward": reward, "valid_sample": True}}
        elif url_path == "/close":
            result = {"closed": True}
        else:
            raise AssertionError(f"Unexpected endpoint {url_path}")
        return _FakeHttpResponse(result)

    agent.server_client.post.side_effect = post
    result = await agent.run(
        SimpleNamespace(cookies={}),
        WebAgentRunRequest(
            responses_create_params={"input": []},
            web_task=WebTask(benchmark="visualwebarena", task_id="0", intent="Find an unavailable item"),
        ),
    )
    assert evaluated == [expected_answer]
    assert result.reward == expected_reward and not result.mask_sample
    judge.assert_not_called()
