# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remote Agent through the agent sessions API, against a scripted remote service."""

import asyncio
import time
from http.cookies import SimpleCookie
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
from nemo_gym.config import AgentServerRef, BaseServerConfig, ResourcesServerRef
from nemo_gym.core import EpisodeId, MaterializedTask, NeMoGymResponseCreateParamsNonStreaming, TaskId
from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_runtime import (
    DirectHTTPToolAccess,
    MCPStreamableHTTPConnection,
    MCPToolAccess,
    SingleAgentTurnRequest,
    SingleAgentTurnTaskInput,
)
from resources_servers.example_single_tool_call.app import (
    SimpleWeatherResourcesServer,
    SimpleWeatherResourcesServerConfig,
)
from responses_api_agents.remote_agent.app import RemoteAgentError
from responses_api_agents.remote_agent.tests.test_app import (
    FakeRemoteResponse,
    fn_call,
    make_agent,
    mock_remote,
    msg,
    traj,
)


TOOLS_URL = "http://resources.test:9000"
WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    "strict": False,
}


class FakeToolResponse:
    """An aiohttp ClientResponse from a resources tool route."""

    def __init__(self, body: dict, set_cookies: dict[str, str] | None = None, status: int = 200) -> None:
        self.status = status
        self._body = orjson.dumps(body)
        self.cookies = SimpleCookie()
        for key, value in (set_cookies or {}).items():
            self.cookies[key] = value
        self.content = SimpleNamespace(read=self.read)

    async def read(self) -> bytes:
        return self._body


def scripted(*turns, tool_handler=None):
    """The remote service plays ``turns`` in order; tool calls to ``TOOLS_URL`` go to ``tool_handler``."""
    received = []
    tool_calls = []

    async def handler(method, url, data=None, headers=None, cookies=None, json=None, **kwargs):
        if url.startswith(TOOLS_URL):
            # Gym's client serializes a json= body before it reaches aiohttp.
            body = orjson.loads(data) if json is None else json
            tool_calls.append({"url": url, "json": body, "cookies": dict(cookies or {}), "headers": dict(headers)})
            return await tool_handler(url, body, cookies)
        received.append({"payload": orjson.loads(data), "cookies": dict(cookies or {})})
        turn = turns[min(len(received) - 1, len(turns) - 1)]
        if isinstance(turn, FakeRemoteResponse):
            return turn
        if callable(turn):
            return await turn()
        return FakeRemoteResponse(200, orjson.dumps(turn))

    request_mock = AsyncMock(side_effect=handler)
    request_mock.received = received
    request_mock.tool_calls = tool_calls
    return request_mock


def seed_request(agent_session_id: str = "agent-session", **overrides) -> AgentSeedSessionRequest:
    fields = {
        "agent_session_id": agent_session_id,
        "episode_id": EpisodeId(rollout_id=f"rollout-{agent_session_id}", attempt=1),
        "task_id": TaskId(taskset="weather", task_id="sf"),
        "tool_accesses": [
            DirectHTTPToolAccess(
                name="weather.direct_http",
                required=True,
                base_url=TOOLS_URL,
                cookies={"resources": "seeded"},
                headers={"x-grant": "1"},
            )
        ],
    }
    return AgentSeedSessionRequest(**(fields | overrides))


ROW = {"input": [{"role": "user", "content": "weather in sf?"}], "tools": [WEATHER_TOOL]}


def activate(client: TestClient, body: dict = ROW):
    return client.post("/v1/responses", json=body, headers={AGENT_ACTIVATION_HEADER: "1"})


class TestSessionActivation:
    def test_tool_calls_go_through_the_grant_and_close_returns_its_cookies(self, monkeypatch) -> None:
        async def weather(url, body, cookies):
            return FakeToolResponse({"city": body["city"], "weather_description": "cold"}, {"resources": "after-tool"})

        service = scripted(
            FakeRemoteResponse(
                200, orjson.dumps(traj([fn_call("c1", "get_weather", '{"city": "SF"}')])), set_cookies={"svc": "1"}
            ),
            traj([msg("It is cold in SF.")]),
            tool_handler=weather,
        )
        mock_remote(monkeypatch, service)
        client = TestClient(make_agent().setup_webserver())

        assert client.post("/v1/agent_sessions", json=seed_request().model_dump(mode="json")).status_code == 200
        reply = activate(client)

        assert reply.status_code == 200, reply.text
        assert [item["type"] for item in reply.json()["output"]] == [
            "function_call",
            "function_call_output",
            "message",
        ]
        assert service.tool_calls == [
            {
                "url": f"{TOOLS_URL}/get_weather",
                "json": {"city": "SF"},
                "cookies": {"resources": "seeded"},
                "headers": {"x-grant": "1", "Content-Type": "application/json"},
            }
        ]
        # The service's cookie round-trips within the activation; the resources cookie never reaches the reply.
        assert service.received[1]["cookies"] == {"svc": "1"}
        assert "resources" not in reply.cookies

        # The same activation again is answered without a second rollout on the service.
        assert activate(client).json() == reply.json()
        assert len(service.received) == 2

        close = client.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": "agent-session", "episode_id": seed_request().episode_id.model_dump()},
        )
        assert close.status_code == 200, close.text
        assert close.json()["resources_cookies"] == {"resources": "after-tool"}
        assert [gap["code"] for gap in close.json()["agent_observations"]["gaps"]] == ["model_calls_unavailable"]

    def test_the_service_cookies_do_not_outlive_the_activation(self, monkeypatch) -> None:
        service = scripted(FakeRemoteResponse(200, orjson.dumps(traj([msg("done")])), set_cookies={"svc": "1"}))
        mock_remote(monkeypatch, service)
        agent = make_agent()
        for session_id in ("first", "second"):
            client = TestClient(agent.setup_webserver())
            client.post("/v1/agent_sessions", json=seed_request(session_id).model_dump(mode="json"))
            assert activate(client).status_code == 200
        assert [call["cookies"] for call in service.received] == [{}, {}]

    def test_an_invalid_responses_object_is_a_final_422(self, monkeypatch) -> None:
        mock_remote(monkeypatch, scripted({"id": "x", "object": "response"}))
        client = TestClient(make_agent().setup_webserver(), raise_server_exceptions=False)
        client.post("/v1/agent_sessions", json=seed_request().model_dump(mode="json"))

        reply = activate(client)

        assert reply.status_code == 422
        assert reply.json()["detail"].startswith("RemoteAgentTerminalError: remote service returned an invalid")

    def test_a_service_error_is_a_retryable_500(self, monkeypatch) -> None:
        mock_remote(monkeypatch, scripted(FakeRemoteResponse(503, b"overloaded")))
        client = TestClient(make_agent().setup_webserver(), raise_server_exceptions=False)
        client.post("/v1/agent_sessions", json=seed_request().model_dump(mode="json"))

        assert activate(client).status_code == 500

    def test_a_tool_call_without_a_grant_is_a_final_422(self, monkeypatch) -> None:
        mock_remote(monkeypatch, scripted(traj([fn_call("c1", "get_weather", '{"city": "SF"}')])))
        client = TestClient(make_agent().setup_webserver(), raise_server_exceptions=False)
        client.post("/v1/agent_sessions", json=seed_request(tool_accesses=[]).model_dump(mode="json"))

        reply = activate(client)

        assert reply.status_code == 422
        assert "no direct HTTP tool access" in reply.json()["detail"]

    def test_run_timeout_secs_bounds_the_activation(self, monkeypatch) -> None:
        async def slow():
            await asyncio.sleep(5)

        mock_remote(monkeypatch, scripted(slow))
        client = TestClient(make_agent(run_timeout_secs=0.05).setup_webserver(), raise_server_exceptions=False)
        client.post("/v1/agent_sessions", json=seed_request().model_dump(mode="json"))

        started = time.monotonic()
        reply = activate(client)

        assert reply.status_code == 500
        assert time.monotonic() - started < 5

    async def test_only_the_activation_deadline_is_reported_as_run_timeout_secs(self, monkeypatch) -> None:
        body = NeMoGymResponseCreateParamsNonStreaming.model_validate(ROW)
        agent = make_agent()
        monkeypatch.setattr(agent, "_run_tool_loop", AsyncMock(side_effect=TimeoutError("inner timeout")))

        with pytest.raises(TimeoutError, match="inner timeout"):
            await agent._activate_session(await agent._seed_agent_session_state(seed_request()), body)

        mock_remote(monkeypatch, scripted(lambda: asyncio.sleep(5)))
        slow = make_agent(run_timeout_secs=0.05)
        with pytest.raises(RemoteAgentError, match="run_timeout_secs=0.05"):
            await slow._activate_session(await slow._seed_agent_session_state(seed_request()), body)


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
            {
                "tool_accesses": [
                    MCPToolAccess(
                        name="search",
                        required=True,
                        connection=MCPStreamableHTTPConnection(url="http://mcp.test/mcp"),
                    )
                ]
            },
            "required MCP tool access: search",
        ),
        (
            {
                "tool_accesses": [
                    DirectHTTPToolAccess(name=name, required=True, base_url=TOOLS_URL) for name in ("a", "b")
                ]
            },
            "at most one direct HTTP tool access",
        ),
    ],
)
def test_unsupported_seeds_are_refused(overrides: dict, reason: str) -> None:
    detail = check_agent_seed_rejected(make_agent().setup_webserver(), seed_request(**overrides))
    assert reason in detail


class _InProcessResponse:
    """A TestClient reply in the shape ServerClient callers read."""

    def __init__(self, reply) -> None:
        self.status = reply.status_code
        self.ok = reply.status_code < 400
        self.headers = reply.headers
        self.cookies = {name: SimpleNamespace(value=value) for name, value in reply.cookies.items()}
        self._body = reply.content
        self.content = SimpleNamespace(read=self.read)

    async def read(self) -> bytes:
        return self._body

    def raise_for_status(self) -> None:
        if not self.ok:
            raise ClientResponseError(MagicMock(), (), status=self.status, message="error")


class _InProcessClient(ServerClient):
    """Routes each server name to an app served in process."""

    model_config = {"arbitrary_types_allowed": True}
    apps: dict

    async def post(self, server_name: str, url_path: str, json=None, cookies=None, headers=None, **kwargs):
        client = self.apps[server_name]
        client.cookies.clear()
        client.cookies.update(dict(cookies or {}))
        body = json.model_dump(mode="json", exclude_unset=True) if hasattr(json, "model_dump") else json
        return _InProcessResponse(client.post(url_path, json=body, headers=headers))

    def _resolve_base_url(self, server_name: str) -> str:
        return TOOLS_URL


async def test_an_environment_server_episode_scores_the_remote_service(monkeypatch) -> None:
    """Seed, activation, close and verify through the real Environment Server and weather Resources Server."""
    resources = SimpleWeatherResourcesServer(
        config=SimpleWeatherResourcesServerConfig(host="resources", port=9000, entrypoint="", name="weather"),
        server_client=MagicMock(spec=ServerClient),
    )
    resources_client = TestClient(resources.setup_webserver())

    async def weather(url, body, cookies):
        resources_client.cookies.clear()
        resources_client.cookies.update(dict(cookies or {}))
        reply = resources_client.post(url.removeprefix(TOOLS_URL), json=body)
        return FakeToolResponse(reply.json(), dict(reply.cookies))

    service = scripted(
        traj([fn_call("c1", "get_weather", '{"city": "SF"}')]),
        traj([msg("It is cold in SF.")]),
        tool_handler=weather,
    )
    mock_remote(monkeypatch, service)
    agent = make_agent(name="remote")
    global_config = OmegaConf.create(
        {
            "weather": {"resources_servers": {"example_single_tool_call": {"host": "resources", "port": 9000}}},
            "remote": {"responses_api_agents": {"remote_agent": {"host": "agent", "port": 8080}}},
        }
    )
    transport = _InProcessClient(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=global_config,
        apps={"weather": resources_client, "remote": TestClient(agent.setup_webserver())},
    )
    environment = SingleAgentTurnEnvironmentServer(
        config=SingleAgentTurnEnvironmentServerConfig(
            name="environment",
            host="environment",
            port=8002,
            entrypoint="app.py",
            resources_server=ResourcesServerRef(type="resources_servers", name="weather"),
            agent_server=AgentServerRef(type="responses_api_agents", name="remote"),
            cleanup_timeout_seconds=10,
            resources_tool_transports=["direct_http"],
        ),
        server_client=transport,
    )

    result = await environment.run_request(
        SingleAgentTurnRequest(
            episode_id=EpisodeId(rollout_id="rollout", attempt=1),
            task=MaterializedTask(
                task_id=TaskId(taskset="weather", task_id="sf"),
                task_input=SingleAgentTurnTaskInput(responses_create_params=ROW, task_data={}),
            ),
        )
    )

    assert result.failure is None, result.failure
    assert result.result.reward == 1.0
    assert service.tool_calls[0]["url"] == f"{TOOLS_URL}/get_weather"
    # The grant carried the resources session the Environment Server seeded.
    assert service.tool_calls[0]["cookies"]
    assert "The weather in SF is cold." in service.received[1]["payload"]["input"][-1]["output"]
    # Close released the session; only its close receipt is kept for a retried close.
    assert all(record.state is None for record in agent._session_records.values())
