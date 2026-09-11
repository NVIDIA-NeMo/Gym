# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.context_management import ContextManagedResponsesClient
from nemo_gym.openai_utils import NeMoGymEasyInputMessage
from nemo_gym.server_utils import ServerClient, get_response_json
from responses_api_agents.simple_agent_with_compaction.app import (
    SimpleAgentWithCompaction,
    SimpleAgentWithCompactionConfig,
    SimpleAgentWithCompactionRunRequest,
)
from responses_api_agents.simple_agent_with_compaction.tests.test_client import (
    answer,
    http_response,
    observation,
    reasoning,
)


def tool_call(arguments='{"action":"move"}'):
    return {"type": "function_call", "id": "function-1", "call_id": "tool-1", "name": "step", "arguments": arguments}


def make_agent(*, responses=None, agent_type=SimpleAgentWithCompaction, config=None, tool_status=200, failure=None):
    server = MagicMock(spec=ServerClient, global_config_dict={"token_id_capture": {"enabled": True}})
    queue = list(responses or [answer(1, output=[tool_call()]), answer(2)])
    calls = []

    async def post(**kwargs):
        calls.append(deepcopy(kwargs))
        path = kwargs["url_path"]
        if path == failure:
            raise ConnectionError(f"lost acknowledgement from {path}")
        if path == "/seed_session":
            return http_response({"image": "seed"}, cookies={"resource_session": "seeded"})
        if path.endswith("/v1/responses"):
            return http_response(queue.pop(0).model_dump(mode="json"), cookies={"model_session": "model"})
        if path == "/step":
            return http_response(
                {"image": "pending", "error": "definite error" if tool_status == 500 else None},
                status=tool_status,
                cookies={"resource_session": "advanced"},
            )
        if path == "/verify":
            return http_response(kwargs["json"] | {"reward": 0.75, "resolved": True})
        if path == "/aggregate_metrics":
            return http_response({"custom_metric": 4})
        raise AssertionError(f"Unexpected HTTP hop: {path}")

    server.post = AsyncMock(side_effect=post)
    values = {
        "host": "127.0.0.1",
        "port": 8080,
        "entrypoint": "",
        "name": "simple",
        "model_server": {"type": "responses_api_models", "name": "policy"},
        "resources_server": {"type": "resources_servers", "name": "resources"},
        "max_steps": 5,
    } | (config or {})
    return agent_type(config=SimpleAgentWithCompactionConfig.model_validate(values), server_client=server), calls


def run_body():
    return SimpleAgentWithCompactionRunRequest.model_validate(
        {
            "responses_create_params": {"input": "task", "max_output_tokens": 8},
            "_ng_rollout_id": "dispatch_g0",
            "task_id": "the-task",
        }
    )


async def run_agent(agent):
    return await agent.run(MagicMock(cookies={"upstream": "kept"}), run_body())


async def test_simple_loop_preserves_tools_reward_final_identity_and_cookies_without_self_http():
    agent, calls = make_agent()
    result = await run_agent(agent)
    assert [call["url_path"] for call in calls] == [
        "/seed_session",
        "/ng-rollout/dispatch_g0_s0/training-token-capture/v1/responses",
        "/step",
        "/ng-rollout/dispatch_g0_s0/training-token-capture/v1/responses",
        "/verify",
    ]
    assert all(call["_retry"] is False for call in calls)
    assert result.reward == 0.75 and result.resolved is True
    assert result.response.id == "response-2"
    assert [item.type for item in result.response.output] == ["function_call", "function_call_output", "message"]
    assert [action.response_id for action in result.context_compaction_result.segments[0].selected_actions] == [
        "response-1",
        "response-2",
    ]
    assert calls[2]["json"] == {"action": "move"}
    assert calls[2]["cookies"] == {"upstream": "kept", "resource_session": "seeded"}
    assert calls[-1]["cookies"] == {"upstream": "kept", "resource_session": "advanced", "model_session": "model"}
    assert "context_compaction_result" not in calls[-1]["json"]
    encoded = result.context_compaction_result.model_dump_json()
    assert "generation_token_ids" not in encoded and "prompt_token_ids" not in encoded


class ImageObservationAgent(SimpleAgentWithCompaction):
    """The only agent-specific adapter is decoding observations; policy stays shared."""

    async def _seed_session_response_messages(self, response):
        return [NeMoGymEasyInputMessage.model_validate(observation((await get_response_json(response))["image"]))]

    async def _tool_response_items(self, output, call_id):
        return [
            *await super()._tool_response_items(output, call_id),
            NeMoGymEasyInputMessage.model_validate(observation(json.loads(output)["image"])),
        ]


async def test_small_agent_hooks_enable_pending_image_compaction_and_keep_verification_semantics():
    agent, calls = make_agent(
        agent_type=ImageObservationAgent,
        config={
            "context_history": {
                "enabled": True,
                "policy": {
                    "type": "recency",
                    "config": {"images": {"enabled": True, "keep_last_groups": 1, "protect_initial_context": False}},
                },
            }
        },
    )
    result = await run_agent(agent)
    assert result.reward == 0.75
    assert [segment.capture_rollout_id for segment in result.context_compaction_result.segments] == [
        "dispatch_g0_s0",
        "dispatch_g0_s1",
    ]
    assert len(result.context_compaction_result.media_assets) == 2
    model_calls = [call for call in calls if call["server_name"] == "policy"]
    assert "seed.png" in model_calls[0]["json"].model_dump_json()
    assert "pending.png" in model_calls[1]["json"].model_dump_json()
    assert "seed.png" not in model_calls[1]["json"].model_dump_json()
    verified = calls[-1]["json"]["response"]["output"]
    assert verified[0]["role"] == "user" and "seed.png" in json.dumps(verified[0])
    assert "pending.png" in json.dumps(verified)
    assert "seed.png" not in result.response.model_dump_json()


def test_verifier_sees_repeated_images_but_transport_exports_each_asset_once(monkeypatch):
    agent, calls = make_agent(
        agent_type=ImageObservationAgent,
        responses=[
            answer(1, output=[tool_call()]),
            answer(2, output=[tool_call() | {"call_id": "tool-2"}]),
            answer(3),
        ],
    )
    body = run_body()
    body.responses_create_params.input = [
        NeMoGymEasyInputMessage.model_validate(observation("initial", "keep this text"))
    ]
    monkeypatch.setattr(type(agent), "get_session_middleware_key", lambda _: "test-secret")
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/run", json=body.model_dump() | {"_ng_rollout_id": "dispatch_g0"})
    assert response.status_code == 200, response.text
    result = response.json()
    verified = json.dumps(calls[-1]["json"]["response"]["output"])
    assert verified.count("https://example.invalid/pending.png") == 2
    assert verified.count("https://example.invalid/seed.png") == 1
    assert "initial.png" in json.dumps(calls[-1]["json"]["responses_create_params"]["input"])
    exported = response.text
    assert exported.count("https://example.invalid/pending.png") == 1
    assert exported.count("https://example.invalid/seed.png") == 1
    assert exported.count("https://example.invalid/initial.png") == 1
    assert "keep this text" in json.dumps(result["responses_create_params"])
    assert "initial.png" not in json.dumps(result["responses_create_params"])
    assert result["reward"] == 0.75


async def test_non_assistant_message_does_not_prematurely_stop_the_simple_agent_loop():
    agent, calls = make_agent(
        responses=[
            answer(1, output=[{"type": "message", "role": "user", "content": "nonterminal"}]),
            answer(2),
        ]
    )
    result = await run_agent(agent)
    assert result.response.id == "response-2"
    assert sum(call["server_name"] == "policy" for call in calls) == 2


@pytest.mark.parametrize("malformed", [True, False])
async def test_definite_tool_errors_are_observations_and_do_not_invalidate_capture(malformed):
    agent, calls = make_agent(
        responses=[answer(1, output=[tool_call("bad-json" if malformed else "{}")]), answer(2)], tool_status=500
    )
    result = await run_agent(agent)
    assert result.context_compaction_result.outcome == "completed"
    observations = [item.output for item in result.response.output if item.type == "function_call_output"]
    assert len(observations) == 1
    assert ("Invalid tool call arguments" if malformed else "definite error") in observations[0]
    assert sum(call["url_path"] == "/step" for call in calls) == (0 if malformed else 1)


@pytest.mark.parametrize("failed_hop", ["/seed_session", "/step"])
async def test_ambiguous_mutation_is_single_attempt_and_never_finishes_a_successful_result(monkeypatch, failed_hop):
    agent, calls = make_agent(failure=failed_hop)
    completed = []
    original = ContextManagedResponsesClient.finish

    def finish(client, *args, **kwargs):
        completed.append(True)
        return original(client, *args, **kwargs)

    monkeypatch.setattr(ContextManagedResponsesClient, "finish", finish)
    with pytest.raises(ConnectionError, match="lost acknowledgement"):
        await run_agent(agent)
    assert sum(call["url_path"] == failed_hop for call in calls) == 1
    assert not completed
    assert all(call["url_path"] != "/verify" for call in calls)


async def test_reasoning_only_response_continues_and_max_steps_is_normal_termination():
    agent, calls = make_agent(
        responses=[answer(1, output=[reasoning(1)]), answer(2, output=[tool_call()])], config={"max_steps": 2}
    )
    result = await run_agent(agent)
    assert result.context_compaction_result.outcome == "max_steps"
    assert result.response.id == "response-2"
    assert len(result.context_compaction_result.segments[0].selected_actions) == 2
    assert sum(call["url_path"] == "/step" for call in calls) == 1


async def test_max_output_tokens_stops_before_tools_and_remains_normal_termination():
    agent, calls = make_agent(responses=[answer(1, output=[tool_call()], incomplete={"reason": "max_output_tokens"})])
    result = await run_agent(agent)
    assert result.context_compaction_result.outcome == "max_output_tokens"
    assert result.context_compaction_result.segments[0].selected_actions[0].finish_reason == "length"
    assert all(call["url_path"] != "/step" for call in calls)


async def test_skip_verification_uses_configured_reward_and_same_capture_result():
    agent, calls = make_agent(config={"skip_verification": True, "skip_verification_reward": 0.3})
    result = await run_agent(agent)
    assert result.reward == 0.3 and result.verification_skipped is True
    assert result.context_compaction_result.outcome == "completed"
    assert all(call["url_path"] != "/verify" for call in calls)


@pytest.mark.parametrize("missing", ["capture", "identity", "invalid_identity"])
async def test_missing_capture_configuration_or_identity_rejects_before_seed_mutation(missing):
    agent, calls = make_agent()
    body = run_body()
    if missing == "capture":
        agent.server_client.global_config_dict = {}
    elif missing == "identity":
        body.capture_rollout_id = None
    else:
        body.capture_rollout_id = "unscoped"
    with pytest.raises(ValueError, match="requires training token capture|framework logical owner"):
        await agent.run(MagicMock(cookies={}), body)
    assert not calls


def test_registered_http_run_returns_compact_result_at_top_level(monkeypatch):
    agent, calls = make_agent(responses=[answer(1)])
    monkeypatch.setattr(type(agent), "get_session_middleware_key", lambda _: "test-secret")
    with TestClient(agent.setup_webserver()) as client:
        response = client.post(
            "/run",
            json={
                "responses_create_params": {"input": "task"},
                "_ng_rollout_id": "dispatch_g0",
            },
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["context_compaction_result"]["logical_rollout_id"] == "dispatch_g0"
    assert "context_compaction_result" not in payload["response"]
    assert payload["response"]["id"] == "response-1"
    assert all(call["_retry"] is False for call in calls)
