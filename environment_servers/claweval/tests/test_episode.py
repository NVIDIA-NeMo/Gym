# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the episode protocol with harness-independent session peers."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from environment_servers.claweval.app import ClawEvalEnvironmentConfig, ClawEvalEnvironmentServer, ClawEvalRequest
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import CleanupContext
from nemo_gym.trajectory import ModelCallRecord, join_agent_model_calls


def assistant_response() -> dict:
    return NeMoGymResponse(
        id="response-1",
        created_at=0,
        model="model",
        object="response",
        status="completed",
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        output=[
            {
                "type": "message",
                "id": "message-1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Your mail is sorted.", "annotations": []}],
            }
        ],
    ).model_dump(mode="json")


class Reply:
    def __init__(self, data: dict, cookies: dict | None = None):
        self.data = data
        self.cookies = {key: SimpleNamespace(value=value) for key, value in (cookies or {}).items()}
        self.status = 200
        self.ok = True
        self.headers = {}

    async def read(self):
        return json.dumps(self.data).encode()

    async def json(self):
        return self.data

    async def text(self):
        return str(self.data)


class Peers:
    def __init__(self, *, failure: str | None = None, max_turns: int = 10):
        self.events = []
        self.turns = 0
        self.failure = failure
        self.max_turns = max_turns

    async def post(self, *, server_name, url_path, json, cookies=None, headers=None):
        self.events.append((url_path, deepcopy(json), deepcopy(cookies), deepcopy(headers)))
        if self.failure == url_path:
            raise RuntimeError("dependency unavailable")
        if url_path == "/seed_session":
            return Reply(
                {
                    "resources_session_id": json["resources_session_id"],
                    "resources_tools": {
                        "server_name": "claweval",
                        "url_path": "/mcp/",
                        "headers": {"Authorization": "Bearer task-token"},
                    },
                },
                {"session": "resource-cookie"},
            )
        if url_path == "/task_info":
            return Reply(
                {
                    "instructions": "Native system prompt",
                    "input": "Sort my mail",
                    "tools": [
                        {
                            "name": "lookup",
                            "description": "Look up a message",
                            "input_schema": {"type": "object", "properties": {}},
                        }
                    ],
                    "max_user_rounds": 1,
                    "timeout_seconds": 5,
                    "max_turns": self.max_turns,
                }
            )
        if url_path == "/v1/agent_sessions":
            return Reply({"agent_session_id": json["agent_session_id"]}, {"session": "agent-cookie"})
        if url_path.endswith("/v1/responses"):
            NeMoGymResponseCreateParamsNonStreaming.model_validate(json)
            self.turns += 1
            response = assistant_response()
            response["usage"] = {
                "input_tokens": self.turns * 10,
                "output_tokens": self.turns,
                "total_tokens": self.turns * 11,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            }
            return Reply(response)
        if url_path == "/simulate_user":
            return Reply({"done": self.turns >= 2, "input": "What about newsletters?"})
        if url_path == "/v1/agent_sessions/close":
            return Reply(
                {
                    "agent_session_id": json["agent_session_id"],
                    "cleanup_confirmed": True,
                    "agent_observations": {
                        "source": "test-harness",
                        "records": [
                            {
                                "kind": "agent_invocation",
                                "invocation_id": "assistant",
                                "model_calls": [{"model_call_id": "first"}, {"model_call_id": "second"}],
                            }
                        ],
                    },
                }
            )
        if url_path == "/verify":
            return Reply({"reward": 0.715, "reward_components": {"completion": 0.8}, "task_score": 71.5})
        if url_path == "/close_session":
            return Reply({"resources_session_id": json["resources_session_id"]})
        raise AssertionError(url_path)


@pytest.fixture
def episode_request():
    return ClawEvalRequest.model_validate(
        {
            "episode_id": {"rollout_id": "episode-1"},
            "task": {
                "task_id": {"taskset": "claweval", "task_id": "T002_email_triage"},
                "task_input": {
                    "task_id": "T002_email_triage",
                    "task_sha256": "a" * 64,
                    "split": "general",
                    "prompt": "Sort my mail",
                },
            },
        }
    )


def environment(peers, transport, sessions):
    client = MagicMock(spec=ServerClient)
    client.post = peers.post
    client.global_config_dict = {"assistant": {"responses_api_agents": {"test": {}}}}
    client._resolve_base_url.return_value = "http://resources:8000"
    config = ClawEvalEnvironmentConfig(
        name="claweval_episode",
        host="127.0.0.1",
        port=8001,
        entrypoint="app.py",
        cleanup_timeout_seconds=5,
        resources_server={"type": "resources_servers", "name": "resources"},
        agent_server={"type": "responses_api_agents", "name": "assistant"},
        tool_transport=transport,
        turn_sessions=sessions,
    )
    return ClawEvalEnvironmentServer(config=config, server_client=client)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport,sessions", [("mcp", "continue"), ("direct_http", "separate")])
async def test_same_task_and_grader_across_harness_session_modes(episode_request, transport, sessions):
    peers = Peers()
    server = environment(peers, transport, sessions)
    cleanup = CleanupContext(episode_request.episode_id, 5)
    try:
        result = await server.run(episode_request, cleanup)
    finally:
        await cleanup.aclose()
    assert result.result.reward == 0.715
    assert result.result.evaluation_completed is True
    assert result.result.response.usage.input_tokens == 30
    assert result.result.response.usage.output_tokens == 3
    assert [item.usage.input_tokens for item in result.result.activation_responses] == [10, 20]
    assert len(result.result.response.output) == 2
    paths = [event[0] for event in peers.events]
    assert paths.count("/verify") == 1
    assert max(i for i, path in enumerate(paths) if path == "/v1/agent_sessions/close") < paths.index("/verify")
    assert paths.index("/verify") < paths.index("/close_session")
    seeds = [event[1] for event in peers.events if event[0] == "/v1/agent_sessions"]
    assert len(seeds) == (1 if sessions == "continue" else 2)
    assert all(seed["tool_accesses"][0]["kind"] == transport for seed in seeds)
    activations = [event for event in peers.events if event[0].endswith("/v1/responses")]
    assert all(event[2] == {"session": "agent-cookie"} for event in activations)
    if sessions == "continue":
        assert [event[3]["x-ng-agent-activation"] for event in activations] == ["1", "2"]
        assert activations[1][1]["input"] == "What about newsletters?"
    else:
        assert len(activations[1][1]["input"]) == 3
    assert all(
        event[2] == {"session": "resource-cookie"}
        for event in peers.events
        if event[0] in {"/task_info", "/simulate_user", "/verify"}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["/v1/agent_sessions", "/ng-rollout/episode-1/v1/responses", "/v1/agent_sessions/close", "/simulate_user"],
)
async def test_dependency_failures_clean_up_without_grading(episode_request, failure):
    peers = Peers(failure=failure)
    server = environment(peers, "mcp", "continue")
    cleanup = CleanupContext(episode_request.episode_id, 5)
    try:
        with pytest.raises(Exception):
            await server.run(episode_request, cleanup)
    finally:
        await cleanup.aclose()
    paths = [event[0] for event in peers.events]
    assert "/verify" not in paths
    assert "/close_session" in paths
    assert "/v1/agent_sessions/close" in paths


@pytest.mark.asyncio
async def test_excess_native_model_turns_never_receive_a_score(episode_request):
    peers = Peers(max_turns=1)
    server = environment(peers, "mcp", "continue")
    cleanup = CleanupContext(episode_request.episode_id, 5)
    try:
        with pytest.raises(Exception, match="model-turn budget"):
            await server.run(episode_request, cleanup)
    finally:
        await cleanup.aclose()
    assert "/verify" not in [event[0] for event in peers.events]


@pytest.mark.asyncio
async def test_unacknowledged_seed_can_close_without_observations(episode_request):
    class UnseededPeers(Peers):
        async def post(self, **kwargs):
            if kwargs["url_path"] == "/v1/agent_sessions/close":
                self.events.append((kwargs["url_path"], kwargs["json"], {}, None))
                return Reply({"agent_session_id": kwargs["json"]["agent_session_id"], "cleanup_confirmed": True})
            return await super().post(**kwargs)

    peers = UnseededPeers(failure="/v1/agent_sessions")
    server = environment(peers, "mcp", "continue")
    cleanup = CleanupContext(episode_request.episode_id, 5)
    with pytest.raises(Exception, match="dependency unavailable"):
        await server.run(episode_request, cleanup)
    await cleanup.aclose()
    paths = [event[0] for event in peers.events]
    assert paths.count("/v1/agent_sessions/close") == 1
    assert "/verify" not in paths


@pytest.mark.asyncio
async def test_persisted_turns_preserve_identity_for_later_capture_join(episode_request):
    class PersistedPeers(Peers):
        async def post(self, **kwargs):
            reply = await super().post(**kwargs)
            if kwargs["url_path"] == "/v1/agent_sessions/close":
                identifier = "globally-unique-harness-invocation"
                reply.data["agent_observations"]["records"] = [
                    {"kind": "agent_invocation", "invocation_id": identifier}
                ]
                reply.data["trajectory"] = {
                    "task_id": "T002_email_triage",
                    "rollout_id": "episode-1",
                    "turns": [
                        {
                            "invocation_id": identifier,
                            "source_message_id": f"assistant-{turn}",
                            "task_id": "T002_email_triage",
                            "rollout_id": "episode-1",
                            "turn_no": turn,
                            "step_count": turn,
                            "timestamp": float(turn),
                        }
                        for turn in (1, 2)
                    ],
                }
            return reply

    peers = PersistedPeers()
    server = environment(peers, "mcp", "continue")
    cleanup = CleanupContext(episode_request.episode_id, 5)
    try:
        result = await server.run(episode_request, cleanup)
    finally:
        await cleanup.aclose()
    assert result.result.ng_trajectory.turns[0].source_message_id == "assistant-1"
    verify = next(body for path, body, _, _ in peers.events if path == "/verify")
    assert verify["verification_input"]["model_turns"] == 2
    joined = join_agent_model_calls(
        result.result.ng_agent_observations,
        [
            ModelCallRecord(
                call_index=0, model_call_id="capture-1", client_session_id="globally-unique-harness-invocation"
            )
        ],
    )
    assert joined.records[0].model_calls[0].model_call_id == "capture-1"
