# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from environment_servers.single_agent_turn.opencode import OpenCodeEnvironmentServer
from responses_api_agents.opencode_multimodal_agent.rewards import OpenCodeToolRewardConfig, score_opencode_tools
from tests.helpers.single_agent_turn import _agent_response, _environment_server, _request, _Response


@pytest.mark.parametrize("explicit_config", [False, True])
@pytest.mark.parametrize("base_reward,auxiliary", [(1.0, 0.1), (0.0, 0.02)])
async def test_session_lifecycle_shapes_reward_and_preserves_training_response(
    base_reward, auxiliary, explicit_config
):
    base, client = _environment_server(token_capture=True)
    config = OpenCodeToolRewardConfig()
    if explicit_config:
        client.global_config_dict.agent.responses_api_agents.test.tool_rewards = config.model_dump()
    call = {"type": "function_call", "call_id": "bash-1", "name": "bash", "arguments": '{"command":"ls"}'}
    tools = score_opencode_tools({"bash-1": ("bash", "completed", 0)}, [call], [], 0.0, config)
    response = _agent_response().model_dump(mode="json")
    response["output"][0]["content"][0]["text"] = "<think>analysis</think>\\boxed{answer}"
    response["opencode_evidence"] = {"turns_used": 2, "finished_naturally": True, "tool_reward": tools.model_dump()}
    client.responses[2] = _Response(response)
    payload = json.loads(client.responses[4].body)
    payload["reward"] = base_reward
    client.responses[4] = _Response(payload)
    server = OpenCodeEnvironmentServer(config=base.config, server_client=client)
    result = await server.run_request(_request())
    assert result.result.reward == pytest.approx(base_reward + auxiliary)
    saved = result.result.model_dump(mode="json")["response"]
    assert saved["output"] == response["output"]
    verify = next(body["json"] for _, path, body in client.calls if path == "/verify")
    assert verify["response"]["output"][0]["content"][0]["text"] == "\\boxed{answer}"
    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/training-token-capture/v1/responses",
        "/v1/agent_sessions/close",
        "/verify",
        "/close_session",
    ]
    assert saved["output"][0]["generation_token_ids"] == [3, 4]


async def test_failed_response_selected_for_verification_is_zero_without_judging_partial_answer():
    from environment_servers.single_agent_turn.app import SessionHandles

    base, client = _environment_server()
    server = OpenCodeEnvironmentServer(config=base.config, server_client=client)
    response = _agent_response().model_copy(
        update={"status": "failed", "error": {"code": "server_error", "message": "limit"}}
    )
    result = await server._verify(
        _request(), SessionHandles(resources_session_id="resource", agent_session_id="agent"), response, None
    )
    assert result.reward == 0.0
    assert result.response.output == response.output
    assert client.calls == []
