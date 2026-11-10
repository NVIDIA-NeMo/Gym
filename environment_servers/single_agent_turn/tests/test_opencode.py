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


@pytest.mark.parametrize("wire_format", ["flat", "native"])
def test_http_bridge_preserves_reward_and_training_tokens(wire_format):
    from fastapi.testclient import TestClient

    base, client = _environment_server(token_capture=True)
    config = OpenCodeToolRewardConfig()
    call = {"type": "function_call", "call_id": "bash-1", "name": "bash", "arguments": '{"command":"ls"}'}
    tools = score_opencode_tools({"bash-1": ("bash", "completed", 0)}, [call], [], 0.0, config)
    response = _agent_response().model_dump(mode="json")
    response["opencode_evidence"] = {"turns_used": 2, "finished_naturally": True, "tool_reward": tools.model_dump()}
    client.responses[2] = _Response(response)
    payload = json.loads(client.responses[4].body)
    payload["reward"] = 1.0
    client.responses[4] = _Response(payload)
    server = OpenCodeEnvironmentServer(config=base.config, server_client=client)
    request = _request()
    if wire_format == "flat":
        task_input = request.task.task_input.model_dump(mode="json")
        body = {
            "responses_create_params": task_input["responses_create_params"],
            **task_input["task_data"],
            "task_source": "agent",
            "_ng_rollout_id": "rollout-a2",
        }
    else:
        body = request.model_dump(mode="json")
    with TestClient(server.setup_webserver()) as http:
        reply = http.post("/run", json=body)
    assert reply.status_code == 200, reply.text
    raw = reply.json()
    result = raw if wire_format == "flat" else raw["result"]
    assert ("episode_id" in raw) is (wire_format == "native")
    assert result["reward"] == pytest.approx(1.1)
    assert result["response"]["output"][0]["generation_token_ids"] == [3, 4]
    assert result["opencode_tool_reward"]["auxiliary_reward"] == pytest.approx(0.1)
    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/training-token-capture/v1/responses",
        "/v1/agent_sessions/close",
        "/verify",
        "/close_session",
    ]


@pytest.mark.parametrize("base_reward", [0.0, 1.0])
async def test_incomplete_response_without_final_tool_evidence_preserves_base_reward_and_tokens(base_reward):
    base, client = _environment_server(token_capture=True)
    response = _agent_response().model_dump(mode="json")
    response["status"] = "incomplete"
    response["incomplete_details"] = {"reason": "max_output_tokens"}
    response["opencode_evidence"] = {"turns_used": 1, "finished_naturally": False, "tool_reward": None}
    client.responses[2] = _Response(response)
    verified = json.loads(client.responses[4].body)
    verified["reward"] = base_reward
    client.responses[4] = _Response(verified)
    server = OpenCodeEnvironmentServer(config=base.config, server_client=client)
    result = await server.run_request(_request())
    assert result.failure is None
    assert result.result.reward == base_reward
    saved = result.result.model_dump(mode="json")
    assert saved["opencode_tool_reward"] is None
    assert saved["finished_naturally"] is False
    assert saved["response"]["output"] == response["output"]
    assert saved["response"]["output"][0]["generation_token_ids"] == [3, 4]
    assert [path for _, path, _ in client.calls][-2:] == ["/verify", "/close_session"]


async def test_completed_response_still_requires_native_evidence_for_shaping():
    from environment_servers.single_agent_turn.app import SessionHandles

    base, client = _environment_server()
    response = _agent_response().model_copy(update={"status": "completed"})
    response.__pydantic_extra__["opencode_evidence"] = {
        "turns_used": 1,
        "finished_naturally": True,
        "tool_reward": None,
    }
    # Direct verification needs only the verifier reply, after the episode has already closed the agent.
    client.responses = [client.responses[4]]
    server = OpenCodeEnvironmentServer(config=base.config, server_client=client)
    with pytest.raises(ValueError, match="Missing native tool evidence"):
        await server._verify(
            _request(), SessionHandles(resources_session_id="resource", agent_session_id="agent"), response, None
        )
