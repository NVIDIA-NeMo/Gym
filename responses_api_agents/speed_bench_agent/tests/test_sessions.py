# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPEED-Bench replay through the agent sessions API."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import orjson
import pytest
from aiohttp import ClientResponseError
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
)
from nemo_gym.agents import AGENT_ACTIVATION_HEADER, AgentSeedSessionRequest, check_agent_seed_rejected
from nemo_gym.client import ServerClient
from nemo_gym.config import AgentServerRef, BaseServerConfig, ModelServerRef, ResourcesServerRef
from nemo_gym.core import EpisodeId, MaterializedTask, TaskId
from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_runtime import (
    DirectHTTPToolAccess,
    MCPStreamableHTTPConnection,
    MCPToolAccess,
    SingleAgentTurnRequest,
    SingleAgentTurnTaskInput,
)
from resources_servers.speed_bench.app import SpeedBenchResourcesServer, SpeedBenchResourcesServerConfig
from responses_api_agents.speed_bench_agent.app import SpeedBenchAgent, SpeedBenchAgentConfig


TWO_TURNS = {"input": [{"role": "user", "content": "first"}, {"role": "user", "content": "second"}]}
SPEC = {
    "num_spec_steps": 2,
    "num_draft_tokens": 4,
    "num_accepted_draft_tokens": 3,
    "num_spec_tokens": 2,
    "acceptance_histogram": [0, 1, 1],
}


def model_reply(text: str) -> dict:
    return {
        "id": f"resp-{text}",
        "created_at": 0.0,
        "model": "policy",
        "object": "response",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "id": f"msg-{text}",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 5,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 8,
        },
        "metrics": {"speculative_decoding": SPEC},
    }


class _Reply:
    """A server reply in the shape ServerClient callers read."""

    def __init__(self, body: bytes, *, status: int = 200, cookies: dict[str, str] | None = None, headers=None):
        self.status = status
        self.ok = status < 400
        self.headers = headers or {}
        self.cookies = {name: SimpleNamespace(value=value) for name, value in (cookies or {}).items()}
        self._body = body
        self.content = SimpleNamespace(read=self.read)

    async def read(self) -> bytes:
        return self._body

    def raise_for_status(self) -> None:
        if not self.ok:
            raise ClientResponseError(MagicMock(), (), status=self.status, message="error")


class _Transport(ServerClient):
    """Serves the policy model from canned replies and every other server from an in-process app."""

    model_config = {"arbitrary_types_allowed": True}
    apps: dict = {}
    model_calls: list = []

    async def post(self, server_name: str, url_path: str, json=None, cookies=None, headers=None, **kwargs):
        body = json.model_dump(mode="json", exclude_unset=True) if hasattr(json, "model_dump") else json
        if server_name == "policy_model":
            self.model_calls.append((url_path, body))
            turn = body["input"][-1]["content"]
            return _Reply(orjson.dumps(model_reply(f"reply to {turn}")), cookies={"model": "m"})
        client = self.apps[server_name]
        client.cookies.clear()
        client.cookies.update(dict(cookies or {}))
        reply = client.post(url_path, json=body, headers=headers)
        return _Reply(reply.content, status=reply.status_code, cookies=dict(reply.cookies), headers=reply.headers)


def make_transport(**apps) -> _Transport:
    return _Transport(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create(
            {
                "speed_bench": {"resources_servers": {"speed_bench": {"host": "resources", "port": 9000}}},
                "agent": {"responses_api_agents": {"speed_bench_agent": {"host": "agent", "port": 8080}}},
            }
        ),
        apps=apps,
        model_calls=[],
    )


def make_agent(transport: ServerClient) -> SpeedBenchAgent:
    return SpeedBenchAgent(
        config=SpeedBenchAgentConfig(
            host="agent",
            port=8080,
            entrypoint="app.py",
            name="agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="speed_bench"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        ),
        server_client=transport,
    )


def seed_request(**overrides) -> AgentSeedSessionRequest:
    fields = {
        "agent_session_id": "agent-session",
        "episode_id": EpisodeId(rollout_id="rollout", attempt=1),
        "task_id": TaskId(taskset="speed_bench", task_id="task"),
    }
    return AgentSeedSessionRequest(**(fields | overrides))


def test_a_session_activation_replays_each_turn_once() -> None:
    transport = make_transport()
    client = TestClient(make_agent(transport).setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_request().model_dump(mode="json")).status_code == 200

    reply = client.post("/ng-rollout/rollout/v1/responses", json=TWO_TURNS, headers={AGENT_ACTIVATION_HEADER: "1"})

    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert [item["content"][0]["text"] for item in body["output"]] == ["reply to first", "reply to second"]
    assert body["usage"]["output_tokens"] == 6
    assert len(body["metrics"]["requests"]) == 2
    assert body["metrics"]["generation_duration"] >= 0
    # Model calls keep the rollout's capture prefix; the session cookie is not handed back.
    assert [path for path, _ in transport.model_calls] == ["/ng-rollout/rollout/v1/responses"] * 2
    assert "model" not in reply.cookies

    # A retried activation gets the replay it already ran, not new timed model calls.
    again = client.post("/ng-rollout/rollout/v1/responses", json=TWO_TURNS, headers={AGENT_ACTIVATION_HEADER: "1"})
    assert again.json() == body
    assert len(transport.model_calls) == 2

    close = client.post(
        "/v1/agent_sessions/close",
        json={"agent_session_id": "agent-session", "episode_id": seed_request().episode_id.model_dump()},
    )
    assert close.status_code == 200, close.text
    # The replay records no invocations, and close says so instead of returning no evidence.
    assert [gap["code"] for gap in close.json()["agent_observations"]["gaps"]] == ["observation_capture_failed"]


def test_a_call_without_a_session_still_propagates_cookies() -> None:
    client = TestClient(make_agent(make_transport()).setup_webserver())
    client.cookies.set("resources", "r")

    reply = client.post("/v1/responses", json=TWO_TURNS)

    assert reply.status_code == 200, reply.text
    assert reply.cookies.get("resources") == "r"
    assert "model" in reply.cookies


def test_optional_tool_access_is_ignored() -> None:
    optional = DirectHTTPToolAccess(name="tools", required=False, base_url="http://tools.test")
    client = TestClient(make_agent(make_transport()).setup_webserver())
    reply = client.post("/v1/agent_sessions", json=seed_request(tool_accesses=[optional]).model_dump(mode="json"))
    assert reply.status_code == 200, reply.text


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (
            {
                "sandbox_access": SandboxAccess(
                    connection=DirectSandboxConnection(provider_config_ref="docker", descriptor={}), workdir="/w"
                )
            },
            "does not support sandbox access",
        ),
        (
            {"tool_accesses": [DirectHTTPToolAccess(name="tools", required=True, base_url="http://tools.test")]},
            "required tool access: tools",
        ),
        (
            {
                "tool_accesses": [
                    MCPToolAccess(
                        name="search", required=True, connection=MCPStreamableHTTPConnection(url="http://mcp.test")
                    )
                ]
            },
            "required tool access: search",
        ),
    ],
)
def test_unsupported_seeds_are_refused(overrides: dict, reason: str) -> None:
    detail = check_agent_seed_rejected(make_agent(make_transport()).setup_webserver(), seed_request(**overrides))
    assert reason in detail


async def test_an_environment_server_episode_keeps_the_replay_timing() -> None:
    """Seed, activation, close and verify through the real Environment Server and SPEED-Bench Resources Server."""
    resources = SpeedBenchResourcesServer(
        config=SpeedBenchResourcesServerConfig(host="resources", port=9000, entrypoint="", name="speed_bench"),
        server_client=MagicMock(spec=ServerClient),
    )
    transport = make_transport(speed_bench=TestClient(resources.setup_webserver()))
    transport.apps["agent"] = TestClient(make_agent(transport).setup_webserver())
    environment = SingleAgentTurnEnvironmentServer(
        config=SingleAgentTurnEnvironmentServerConfig(
            name="environment",
            host="environment",
            port=8002,
            entrypoint="app.py",
            resources_server=ResourcesServerRef(type="resources_servers", name="speed_bench"),
            agent_server=AgentServerRef(type="responses_api_agents", name="agent"),
            cleanup_timeout_seconds=10,
        ),
        server_client=transport,
    )

    result = await environment.run_request(
        SingleAgentTurnRequest(
            episode_id=EpisodeId(rollout_id="rollout", attempt=1),
            task=MaterializedTask(
                task_id=TaskId(taskset="speed_bench", task_id="task"),
                task_input=SingleAgentTurnTaskInput(responses_create_params=TWO_TURNS, task_data={}),
            ),
        )
    )

    assert result.failure is None, result.failure
    verified = result.result.model_dump()
    timing = verified["response"]["metrics"]
    # The verifier reads the replay's own window, which the session round trips do not touch.
    assert verified["gen_seconds"] == timing["generation_duration"]
    assert verified["generation_started_at"] == timing["generation_started_at"]
    assert verified["num_generated_tokens"] == 6
    assert verified["acceptance_length"] == pytest.approx(2.5)
    assert verified["spec_decode_unavailable"] is False
