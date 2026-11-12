# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent sessions: seed, one activation, close, and how a mini-SWE run becomes a response."""

import asyncio
import dataclasses
import json
import sys
import typing
from contextlib import AsyncExitStack, asynccontextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    ROLLOUT_STOP_METADATA_KEY,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    SandboxSession,
    check_agent_seed_rejected,
)
from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, NeMoGymResponseCreateParamsNonStreaming, TaskId
from nemo_gym.data.tasks.harbor import HarborMCPServer
from nemo_gym.server_runtime import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess
from responses_api_agents.mini_swe_agent import app as app_module
from responses_api_agents.mini_swe_agent import mcp_cli
from responses_api_agents.mini_swe_agent.app import MiniSWEAgent, MiniSWEAgentConfig
from responses_api_agents.mini_swe_agent.sandbox import HarnessProcessInfo, MiniSWEArtifacts, MiniSWESession
from responses_api_agents.mini_swe_agent.tests.test_trajectory import model_response, native_trajectory


SANDBOX_ACCESS = {
    "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {"sandbox_id": "task-box"}},
    "workdir": "/app",
}
# The real read, before the sandbox fixture stubs it.
READ_TASK_CONTEXT = MiniSWESession.read_task_context


def config(**overrides) -> MiniSWEAgentConfig:
    return MiniSWEAgentConfig(
        name="mini_swe_agent",
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        model_server={"type": "responses_api_models", "name": "policy_model"},
        model="served-model",
        concurrency=4,
        step_limit=500,
        step_timeout=30,
        **overrides,
    )


def seed(**overrides) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="session",
        episode_id=EpisodeId(rollout_id="rollout"),
        task_id=TaskId(taskset="test", task_id="task"),
        **({"sandbox_access": SANDBOX_ACCESS} | overrides),
    )


def mcp_grant(name: str = "task_tools", *, required: bool = True) -> MCPToolAccess:
    return MCPToolAccess(
        name=name,
        required=required,
        connection=MCPStreamableHTTPConnection(url="http://10.0.0.5:9000/mcp", headers={"Authorization": "Bearer t"}),
    )


@pytest.fixture
def agent(monkeypatch) -> MiniSWEAgent:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    monkeypatch.setattr(
        MiniSWEAgent,
        "resolve_model_base_url",
        lambda self, name, rollout_id=None, **kwargs: f"http://10.0.0.5:8000/ng-rollout/{rollout_id}/v1",
    )
    monkeypatch.setattr(app_module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app_module, "resolve_provider_config", lambda ref, global_config: {"docker": {}})
    monkeypatch.setattr(app_module, "resolve_provider_metadata", lambda ref, global_config: {"team": "gym"})
    return MiniSWEAgent(config=config(), server_client=server_client)


@pytest.fixture
def sandbox(monkeypatch) -> SimpleNamespace:
    """A sandbox the agent connects to or starts; install and the task-context read are stubbed per test."""
    provider = SimpleNamespace(aclose=AsyncMock())
    box = SimpleNamespace(
        sandbox_id="task-box",
        start=AsyncMock(),
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None)),
    )
    connect = AsyncMock(return_value=box)
    monkeypatch.setattr(app_module, "create_provider", lambda provider_config: provider)
    monkeypatch.setattr(app_module.AsyncSandbox, "connect", connect)
    monkeypatch.setattr(app_module, "AsyncSandbox", MagicMock(return_value=box, connect=connect))
    install = AsyncMock()
    task_context = AsyncMock(return_value={})
    monkeypatch.setattr(MiniSWESession, "install", install)
    monkeypatch.setattr(MiniSWESession, "read_task_context", task_context)
    return SimpleNamespace(box=box, provider=provider, connect=connect, install=install, task_context=task_context)


def state_for(agent: MiniSWEAgent, request: AgentSeedSessionRequest | None = None) -> MiniSWESession:
    box = SimpleNamespace(sandbox_id="task-box", exec=AsyncMock())
    return MiniSWESession(
        request=request or seed(),
        provider_name="sandbox",
        sandbox_id="task-box",
        session=SandboxSession(
            sandbox=box, workdir="/app", session_dir="/tmp/nemo-gym-mini-swe-sessions/s", harness="mini-SWE"
        ),
    )


def http_request(state: MiniSWESession, rollout_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        session={"agent_session_id": state.request.agent_session_id},
        path_params={"rollout_id": rollout_id or state.request.episode_id.capture_key},
    )


def bind(agent: MiniSWEAgent, state: MiniSWESession) -> None:
    from nemo_gym.agents.base_responses_api_agent import _AgentSessionRecord

    agent._session_records[state.request.agent_session_id] = _AgentSessionRecord(
        state=state, episode_id=state.request.episode_id
    )


def finish_run(
    state: MiniSWESession,
    *,
    trajectory: dict | None,
    return_code: int = 0,
    timed_out: bool = False,
    output: dict | None = None,
    mcp_calls: list | None = None,
    invalid_mcp_calls: int = 0,
) -> AsyncMock:
    """Make the session's execute end as a run that left these artifacts and this supervisor receipt."""
    artifacts = MiniSWEArtifacts(
        trajectory=trajectory,
        output=output if output is not None else {"runtime": {"hostname": "box", "pid": 7, "python": "/py"}},
        mcp_calls=mcp_calls or [],
        log="",
        invalid_mcp_calls=invalid_mcp_calls,
    )

    async def execute(*, stage_activation, collect, timeout, close_timeout):
        state.session.launch_started = True
        state.session.cleanup = {
            "cleanup_confirmed": True,
            "return_code": return_code,
            "timed_out": timed_out,
            "error": None,
        }
        state.session.artifacts = artifacts
        state.runtime_info = HarnessProcessInfo(hostname="box", pid=7, python="/py")
        return artifacts

    state.session.execute = AsyncMock(side_effect=execute)
    return state.session.execute


def body(**overrides) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(**{"input": "Fix the failing test.", **overrides})


# -- seed ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"sandbox_access": None}, "sandbox_access or a configured sandbox_provider"),
        ({"sandbox_access": SANDBOX_ACCESS | {"user": "agent"}}, "task user is unsupported"),
        ({"sandbox_access": SANDBOX_ACCESS | {"workdir": "app"}}, "absolute task workdir"),
        ({"sandbox_access": SANDBOX_ACCESS | {"workdir": "/tmp"}}, "disjoint from its session directory"),
        (
            {"sandbox_access": SANDBOX_ACCESS | {"workdir": "/tmp/nemo-gym-mini-swe-sessions/x"}},
            "disjoint from its session directory",
        ),
        (
            {"tool_accesses": [DirectHTTPToolAccess(name="tools", required=True, base_url="http://r:1")]},
            "supports only MCP tool grants",
        ),
    ],
)
def test_unsupported_seeds_are_final_refusals(agent, sandbox, overrides, reason) -> None:
    detail = check_agent_seed_rejected(agent.setup_webserver(), seed(**overrides))

    assert reason in detail
    assert not agent._session_records
    sandbox.connect.assert_not_awaited()


def test_a_loopback_model_url_is_refused_for_a_remote_sandbox(agent, sandbox, monkeypatch) -> None:
    monkeypatch.setattr(
        MiniSWEAgent, "resolve_model_base_url", lambda *args, **kwargs: "http://127.0.0.1:8000/ng-rollout/r/v1"
    )
    monkeypatch.setattr(app_module, "resolve_provider_config", lambda ref, global_config: {"opensandbox": {}})

    assert "loopback" in check_agent_seed_rejected(agent.setup_webserver(), seed())


async def test_a_borrowed_seed_connects_installs_and_reads_the_task_tools(agent, sandbox) -> None:
    sandbox.task_context.return_value = {
        "mcp_servers": [{"name": "browser", "transport": "streamable_http", "url": "http://localhost:3000/mcp"}],
        "skills_dir": "/skills",
    }

    state = await agent._seed_agent_session_state(seed(tool_accesses=[mcp_grant(required=False)]))

    sandbox.connect.assert_awaited_once()
    sandbox.box.start.assert_not_awaited()
    sandbox.install.assert_awaited_once_with(timeout=900)
    assert not state.session.owns_sandbox
    assert state.session.workdir == "/app"
    assert state.session.session_dir.startswith("/tmp/nemo-gym-mini-swe-sessions/")
    # Grants come first with their headers; the task's own servers are required by the task.
    assert [server["name"] for server in state.mcp_servers] == ["task_tools", "browser"]
    assert state.mcp_servers[0]["headers"] == {"Authorization": "Bearer t"}
    assert state.required_mcp_servers == ["browser"]
    assert state.task_context["skills_dir"] == "/skills"


async def test_an_owned_seed_starts_a_sandbox_with_a_ttl_and_never_reads_task_context(agent, sandbox) -> None:
    agent.config.sandbox_provider = "sandbox"
    agent.config.sandbox_config = {"image": "python:3.13-slim", "metadata": {"run": "smoke"}}

    state = await agent._seed_agent_session_state(seed(sandbox_access=None))

    (spec,), _ = sandbox.box.start.await_args
    assert spec.ttl_s == 900 + 28800 + 600
    assert spec.workdir == "/app"
    assert spec.metadata == {"team": "gym", "run": "smoke", "agent_session_id": "session"}
    assert state.session.owns_sandbox
    sandbox.task_context.assert_not_awaited()


async def test_an_owned_seed_serves_the_episodes_mcp_grants(agent, sandbox) -> None:
    agent.config.sandbox_provider = "sandbox"

    state = await asyncio.wait_for(
        agent._seed_agent_session_state(seed(sandbox_access=None, tool_accesses=[mcp_grant(required=True)])), timeout=5
    )

    assert state.session.owns_sandbox
    assert [(server["name"], server["headers"]) for server in state.mcp_servers] == [
        ("task_tools", {"Authorization": "Bearer t"})
    ]
    assert state.required_mcp_servers == ["task_tools"]
    sandbox.task_context.assert_not_awaited()


@pytest.mark.parametrize(
    ("task_json", "reason"),
    [
        ("{oops", "is not JSON"),
        ("[1]", "is not a JSON object"),
        ('{"mcp_servers": [{"transport": "sse", "url": "http://tools/sse"}]}', "MCP server without a name"),
    ],
)
def test_task_data_errors_are_final_refusals(agent, sandbox, monkeypatch, task_json, reason) -> None:
    monkeypatch.setattr(MiniSWESession, "read_task_context", READ_TASK_CONTEXT)
    close = AsyncMock()
    monkeypatch.setattr(MiniSWESession, "close", close)
    sandbox.box.exec.return_value = SimpleNamespace(return_code=0, stdout=task_json, stderr="", error_type=None)

    # The same task data fails every attempt, so the seed is refused instead of retried.
    assert reason in check_agent_seed_rejected(agent.setup_webserver(), seed())
    close.assert_awaited_once()


def fake_mcp_sdk(monkeypatch) -> list[tuple]:
    """Stand-ins for the runtime's MCP 2 SDK that record which client each server reached."""
    opened: list[tuple] = []

    @asynccontextmanager
    async def streams(*key):
        opened.append(key)
        yield "read", "write"

    class ClientSession:
        def __init__(self, read, write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def initialize(self):
            pass

    class AsyncClient:
        def __init__(self, *, headers, timeout):
            self.headers = headers

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    modules = {
        "httpx2": SimpleNamespace(AsyncClient=AsyncClient, Timeout=lambda *args, **kwargs: None),
        "mcp": SimpleNamespace(ClientSession=ClientSession, StdioServerParameters=SimpleNamespace),
        "mcp.client.sse": SimpleNamespace(sse_client=lambda url, headers=None: streams("sse", url, headers)),
        "mcp.client.stdio": SimpleNamespace(
            stdio_client=lambda parameters: streams("stdio", parameters.command, tuple(parameters.args))
        ),
        "mcp.client.streamable_http": SimpleNamespace(
            streamable_http_client=lambda url, http_client: streams("streamable_http", url, http_client.headers)
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "mcp.client", ModuleType("mcp.client"))
    return opened


@pytest.mark.parametrize("transport", typing.get_args(HarborMCPServer.model_fields["transport"].annotation))
async def test_each_harbor_task_server_transport_reaches_its_mcp_client(
    agent, sandbox, monkeypatch, transport
) -> None:
    # The file the Harbor environment server writes: HarborMCPServer dumps under mcp_servers.
    declared = HarborMCPServer(
        name="task_tools", transport=transport, url="http://tools:8000/mcp", command="serve-tools", args=["--fast"]
    )
    sandbox.task_context.return_value = {"mcp_servers": [declared.model_dump(mode="json")], "skills_dir": None}
    state = await asyncio.wait_for(agent._seed_agent_session_state(seed()), timeout=5)
    (server,) = state.mcp_servers
    opened = fake_mcp_sdk(monkeypatch)

    async with AsyncExitStack() as stack:
        await mcp_cli.connect(server, stack)

    expected = {
        "stdio": ("stdio", "serve-tools", ("--fast",)),
        "sse": ("sse", "http://tools:8000/mcp", None),
        "streamable-http": ("streamable_http", "http://tools:8000/mcp", None),
    }
    assert opened == [expected[transport]]
    assert state.required_mcp_servers == ["task_tools"]


async def test_a_task_server_that_harbor_also_grants_is_connected_once_and_required(agent, sandbox) -> None:
    sandbox.task_context.return_value = {
        "mcp_servers": [{"name": "task_tools", "transport": "streamable-http", "url": "http://x/mcp"}]
    }

    state = await agent._seed_agent_session_state(seed(tool_accesses=[mcp_grant(required=False)]))

    assert [server["name"] for server in state.mcp_servers] == ["task_tools"]
    assert state.mcp_servers[0]["headers"] == {"Authorization": "Bearer t"}
    assert state.required_mcp_servers == ["task_tools"]


async def test_mcp_server_names_must_not_collide(agent, sandbox, monkeypatch) -> None:
    sandbox.task_context.return_value = {"mcp_servers": [{"name": "browser", "url": "a"}, {"name": "browser"}]}
    close = AsyncMock()
    monkeypatch.setattr(MiniSWESession, "close", close)

    with pytest.raises(AgentSeedRejectedError, match="MCP server names collide: browser"):
        await agent._seed_agent_session_state(seed())
    # The refusal comes after the sandbox was set up, so the setup is released before the seed fails.
    close.assert_awaited_once()


async def test_a_failed_install_releases_the_sandbox_and_raises(agent, sandbox, monkeypatch) -> None:
    sandbox.install.side_effect = RuntimeError("Installing the mini-swe Python runtime failed with 1")
    close = AsyncMock()
    monkeypatch.setattr(MiniSWESession, "close", close)

    with pytest.raises(RuntimeError, match="runtime failed"):
        await agent._seed_agent_session_state(seed())
    close.assert_awaited_once()


async def test_a_failed_install_whose_cleanup_fails_is_kept_for_close(agent, sandbox, monkeypatch) -> None:
    sandbox.install.side_effect = RuntimeError("install failed")
    monkeypatch.setattr(MiniSWESession, "close", AsyncMock(side_effect=RuntimeError("release failed")))

    with pytest.raises(AgentSessionSetupError) as error:
        await agent._seed_agent_session_state(seed())
    assert isinstance(error.value.state, MiniSWESession)
    assert str(error.value.error) == "install failed"


async def test_a_failed_connect_closes_the_provider(agent, sandbox) -> None:
    sandbox.connect.side_effect = ConnectionError("sandbox gone")

    with pytest.raises(ConnectionError):
        await agent._seed_agent_session_state(seed())
    sandbox.provider.aclose.assert_awaited_once()


# -- activation ------------------------------------------------------------------------------------------------------


async def test_one_activation_runs_once_and_identical_retries_join_it(agent) -> None:
    state = state_for(agent)
    bind(agent, state)
    execute = finish_run(state, trajectory=native_trajectory())

    first, second = await asyncio.wait_for(
        asyncio.gather(agent.responses(http_request(state), body()), agent.responses(http_request(state), body())),
        timeout=5,
    )

    assert execute.await_count == 1
    assert first.id == second.id and first.status == "completed"
    with pytest.raises(HTTPException) as conflict:
        await agent.responses(http_request(state), body(temperature=0.1))
    assert conflict.value.status_code == 409


async def test_activation_must_use_the_seeded_rollout_route(agent) -> None:
    state = state_for(agent)
    bind(agent, state)

    with pytest.raises(HTTPException) as error:
        await agent.responses(http_request(state, rollout_id="other-1"), body())
    assert error.value.status_code == 409


async def test_unknown_session_is_a_conflict_not_a_host_run(agent) -> None:
    with pytest.raises(HTTPException) as error:
        await agent.responses(SimpleNamespace(session={"agent_session_id": "nope"}, path_params={}), body())
    assert error.value.status_code == 409


async def test_request_tools_need_an_mcp_server_to_reach_them(agent) -> None:
    state = state_for(agent)
    bind(agent, state)
    finish_run(state, trajectory=native_trajectory())
    tools = [{"type": "function", "name": "get_weather", "parameters": {}, "strict": False}]

    # Without one, the episode would run without its tools and score a silent zero.
    with pytest.raises(HTTPException, match="only through MCP grants") as error:
        await agent.responses(http_request(state), body(tools=tools))
    assert error.value.status_code == 422 and state.task is None

    state.mcp_servers = [{"name": "task_tools"}]
    response = await asyncio.wait_for(agent.responses(http_request(state), body(tools=tools)), timeout=5)
    assert response.status == "completed"


async def test_a_disconnected_waiter_does_not_cancel_the_shared_activation(agent) -> None:
    state = state_for(agent)
    bind(agent, state)
    execute = finish_run(state, trajectory=native_trajectory())
    inner, started, release = execute.side_effect, asyncio.Event(), asyncio.Event()

    async def slow(**kwargs):
        started.set()
        await release.wait()
        return await inner(**kwargs)

    execute.side_effect = slow
    first = asyncio.create_task(agent.responses(http_request(state), body()))
    await asyncio.wait_for(started.wait(), timeout=5)

    first.cancel()  # The Environment Server's request was dropped.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(first, timeout=5)
    assert not state.task.done()
    release.set()
    retried = await asyncio.wait_for(agent.responses(http_request(state), body()), timeout=5)

    assert retried.status == "completed" and execute.await_count == 1


def test_the_http_session_lifecycle_runs_once_and_refuses_activation_after_close(agent, sandbox, monkeypatch) -> None:
    runs = []

    async def execute(self, *, stage_activation, collect, timeout, close_timeout):
        runs.append(self.session_dir)
        self.launch_started = True
        self.cleanup = {"cleanup_confirmed": True, "return_code": 0, "timed_out": False, "error": None}
        self.artifacts = MiniSWEArtifacts(
            trajectory=native_trajectory(), output={"runtime": {"hostname": "box", "pid": 7}}, mcp_calls=[], log=""
        )
        return self.artifacts

    monkeypatch.setattr(SandboxSession, "execute", execute)
    monkeypatch.setattr(SandboxSession, "close", AsyncMock())
    request = seed()
    rollout = f"/ng-rollout/{request.episode_id.capture_key}/v1/responses"
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=request.model_dump(mode="json"))
        assert created.status_code == 200, created.text
        session_id = created.json()["agent_session_id"]
        assert client.post("/v1/responses", json={"input": "task"}).status_code == 409
        assert client.post("/ng-rollout/other-1/v1/responses", json={"input": "task"}).status_code == 409
        refused = client.post(rollout, json={"input": "task", "tool_choice": "none"})
        assert refused.status_code == 422 and "tool_choice" in refused.text
        state = agent._session_records[session_id].state
        assert state.task is None and state.activation_request is None
        assert not runs
        # A rejected request did not bind the session; a per-call token limit is supported.
        activation = {"input": "task", "max_output_tokens": 8}
        first = client.post(rollout, json=activation)
        assert first.status_code == 200, first.text
        assert first.json()["status"] == "completed"
        assert client.post(rollout, json=activation).json() == first.json()
        assert client.post(rollout, json={**activation, "max_output_tokens": 16}).status_code == 409
        assert client.post(rollout, json={"input": "changed"}).status_code == 409
        closed = client.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": session_id, "episode_id": request.episode_id.model_dump(mode="json")},
        )
        assert closed.status_code == 200, closed.text
        assert closed.json()["cleanup_confirmed"] is True
        assert closed.json()["agent_observations"]["source"] == "mini_swe"
        assert closed.json().get("resources_cookies") is None
        assert client.post(rollout, json={"input": "task"}).status_code == 409
    assert len(runs) == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": "other-model"}, "must match the configured model"),
        ({"previous_response_id": "resp_1"}, "request field previous_response_id"),
        ({"store": True}, "request field store"),
        ({"metadata": {"agent_timeout_sec": "60", "label": "x"}}, "only agent_timeout_sec"),
        ({"input": "   "}, "nonempty task instruction"),
        ({"input": [{"type": "function_call_output", "call_id": "c", "output": "x"}]}, "message input only"),
    ],
)
def test_requests_the_harness_cannot_honor_are_rejected(agent, overrides, message) -> None:
    with pytest.raises(HTTPException, match=message) as error:
        agent._validate_request(NeMoGymResponseCreateParamsNonStreaming(**{"input": "task", **overrides}))
    assert error.value.status_code == 422


@pytest.mark.parametrize("token_cap", [None, 1, 8, 32])
def test_the_payload_routes_every_model_call_through_the_rollout_and_forwards_sampling(agent, token_cap) -> None:
    request = seed(sandbox_access=SANDBOX_ACCESS | {"env": {"CIRCLE_NODE_TOTAL": "3"}})
    state = state_for(agent, request)
    state.task_context = {"skills_dir": "/skills"}
    state.mcp_servers = [{"name": "task_tools"}]
    state.required_mcp_servers = ["task_tools"]
    params = body(
        input=[
            {"role": "developer", "content": "You are helpful."},
            {"role": "user", "content": [{"type": "input_text", "text": "What's the weather?"}]},
        ],
        temperature=0.6,
        top_p=0.9,
        max_output_tokens=token_cap,
        metadata={"agent_timeout_sec": "60"},
        tools=[{"type": "function", "name": "get_weather", "parameters": {}, "strict": False}],
    )

    payload = agent._activation_payload(state, params, timeout=60)

    assert payload["task"] == "You are helpful.\n\nWhat's the weather?"
    assert payload["skills_dir"] == "/skills"
    assert payload["mcp"] == {"servers": ["task_tools"], "required": ["task_tools"]}
    assert payload["config"]["agent"] == {"step_limit": 500, "cost_limit": 0}
    # The command timeout never outlasts the run.
    assert payload["config"]["environment"] == {"cwd": "/app", "timeout": 30, "env": {"CIRCLE_NODE_TOTAL": "3"}}
    model = payload["config"]["model"]
    assert model["model_name"] == "openai/served-model"
    kwargs = model["model_kwargs"]
    assert kwargs["api_base"] == "http://10.0.0.5:8000/ng-rollout/rollout/v1"
    assert kwargs["extra_headers"] == {"x-session-id": "rollout"}
    if token_cap is None:
        assert "extra_body" not in kwargs
    else:
        assert kwargs["extra_body"] == {"max_output_tokens": token_cap}
    assert (kwargs["temperature"], kwargs["top_p"], kwargs["max_retries"]) == (0.6, 0.9, 0)
    # Only mini-SWE's bash tool reaches the model; the request's tools and metadata are not model arguments.
    assert "tools" not in kwargs and "metadata" not in kwargs and "max_output_tokens" not in kwargs


@pytest.mark.parametrize(
    ("seed_budget", "metadata", "expected"),
    [
        (None, None, 28800),
        (600, None, 600),
        (600, {"agent_timeout_sec": "60"}, 60),
        (None, {"agent_timeout_sec": "x"}, 28800),
    ],
)
def test_the_run_budget_is_the_ceiling_shortened_by_the_task(agent, seed_budget, metadata, expected) -> None:
    state = state_for(agent, seed(agent_timeout_sec=seed_budget))

    assert agent._agent_timeout_sec(state, body(metadata=metadata)) == expected


def test_the_command_timeout_never_outlasts_the_run(agent) -> None:
    state = state_for(agent)

    assert agent._activation_payload(state, body(), timeout=10)["config"]["environment"]["timeout"] == 10
    assert agent._activation_payload(state, body(), timeout=0.2)["config"]["environment"]["timeout"] == 1


async def test_a_submitted_run_is_completed_with_usage_and_observations(agent) -> None:
    state = state_for(agent)
    finish_run(state, trajectory=native_trajectory())

    response = await agent._activate(state, body())

    assert response.status == "completed"
    assert response.usage.total_tokens == 45
    assert response.metadata["mini_swe_termination"] == "completed"
    assert response.metadata["harness_pid"] == "7"
    assert AGENT_OUTCOME_METADATA_KEY not in response.metadata
    bundle = state.observations
    invocation, *tools, sandbox_record = bundle.records
    assert (invocation.invocation_id, invocation.status) == ("rollout", "completed")
    assert len(invocation.model_calls) == 3
    assert [tool.tool_call_id for tool in tools] == ["call_1", "call_2"]
    assert (sandbox_record.outcome, sandbox_record.exit_code, sandbox_record.error_type) == ("completed", 0, None)


@pytest.mark.parametrize("exit_status", ["LimitsExceeded", "ContextWindowExceededError"])
async def test_a_limit_stop_is_incomplete_and_verifiable(agent, exit_status) -> None:
    trajectory = native_trajectory()
    trajectory["info"]["exit_status"] = exit_status
    state = state_for(agent)
    finish_run(state, trajectory=trajectory)

    response = await agent._activate(state, body())

    assert response.status == "incomplete"
    assert response.metadata["mini_swe_termination"] == exit_status


async def test_a_deadline_stop_is_incomplete_with_partial_work(agent) -> None:
    trajectory = native_trajectory()
    trajectory["info"]["exit_status"] = ""
    del trajectory["messages"][-1]
    state = state_for(agent)
    finish_run(state, trajectory=trajectory, return_code=143, timed_out=True)

    response = await agent._activate(state, body())

    assert response.status == "incomplete"
    assert response.metadata["stop_reason"] == "wall_time"
    assert [item.type for item in response.output][-1] == "function_call"
    invocation = state.observations.records[0]
    assert invocation.stop_reason == "wall_time"
    assert "native_trajectory_partial" in [gap.code for gap in state.observations.gaps]
    assert state.observations.records[-1].outcome == "timeout"


@pytest.mark.parametrize(
    "trajectory",
    [None, {"info": {"exit_status": ""}, "messages": native_trajectory()["messages"][:2]}],
    ids=["before-the-first-save", "system-and-task-only"],
)
async def test_a_deadline_without_a_model_reply_keeps_its_cause_unknown(agent, trajectory) -> None:
    state = state_for(agent)
    finish_run(state, trajectory=trajectory, return_code=143, timed_out=True)

    response = await asyncio.wait_for(agent._activate(state, body()), timeout=5)

    # This also covers a runner stalled during startup, before making any model request.
    assert (response.status, response.metadata["mini_swe_termination"]) == ("failed", "AgentDeadline")
    assert json.loads(response.metadata[AGENT_OUTCOME_METADATA_KEY]) == {"failure_kind": "unknown"}
    assert response.metadata["stop_reason"] == "wall_time"
    assert "no model call returned" in response.error.message
    invocation = state.observations.records[0]
    assert (invocation.status, invocation.stop_reason) == ("failed", "wall_time")


async def test_a_rollout_stop_before_the_first_save_is_incomplete_and_marked(agent) -> None:
    state = state_for(agent)
    execute = finish_run(state, trajectory=None, return_code=None, timed_out=True, output={})
    inner = execute.side_effect

    async def expired(**kwargs):
        artifacts = await inner(**kwargs)
        # What SandboxSession.expire leaves behind when POST /rollouts/<key>/stop ended the run.
        state.session._expire_requested, state.session.expire_reason = True, "operator"
        state.runtime_info = None
        return artifacts

    execute.side_effect = expired

    response = await asyncio.wait_for(agent._activate(state, body()), timeout=5)

    assert (response.status, response.output) == ("incomplete", [])
    assert response.metadata["mini_swe_termination"] == "timeout"
    assert AGENT_OUTCOME_METADATA_KEY not in response.metadata
    assert json.loads(response.metadata[ROLLOUT_STOP_METADATA_KEY]) == {"reason": "operator"}
    gaps = [gap.code for gap in state.observations.gaps]
    assert "native_trajectory_unavailable" in gaps and "runtime_info_unavailable" in gaps
    assert "usage_unavailable" not in gaps
    assert state.observations.records[-1].outcome == "timeout"


async def test_usage_stays_unknown_on_the_response_when_one_call_reported_none(agent) -> None:
    trajectory = native_trajectory()
    trajectory["messages"][4]["extra"]["response"] = model_response("resp_rejected", usage=False)
    state = state_for(agent)
    finish_run(state, trajectory=trajectory)

    response = await asyncio.wait_for(agent._activate(state, body()), timeout=5)

    assert response.status == "completed" and response.usage is None
    assert "usage_unavailable" in [gap.code for gap in state.observations.gaps]


async def test_a_saved_response_that_is_not_a_responses_object_is_retryable(agent) -> None:
    trajectory = native_trajectory()
    trajectory["messages"][5]["output"] = "not a list"
    state = state_for(agent)
    finish_run(state, trajectory=trajectory)

    with pytest.raises(RuntimeError, match="not a valid Responses API object"):
        await asyncio.wait_for(agent._activate(state, body()), timeout=5)


async def test_a_run_killed_mid_step_is_the_agents_own_result(agent) -> None:
    trajectory = native_trajectory()
    trajectory["info"]["exit_status"] = ""
    state = state_for(agent)
    finish_run(state, trajectory=trajectory, return_code=137)

    response = await agent._activate(state, body())

    assert (response.status, response.metadata["mini_swe_termination"]) == ("incomplete", "killed")


async def test_a_run_close_stopped_is_cancelled_when_its_activation_returns_before_close_cancels_it(agent) -> None:
    trajectory = native_trajectory()
    trajectory["info"]["exit_status"] = ""
    # The submission command was running when close stopped the harness: mini-SWE saved no exit message.
    del trajectory["messages"][-1]
    state = state_for(agent)
    # What sandbox_runner.py writes when the supervisor's SIGTERM follows the stop request close wrote.
    output = {"runtime": {"hostname": "box", "pid": 7}, "stop_reason": "cancelled", "error": None, "mcp": None}
    finish_run(state, trajectory=trajectory, return_code=143, output=output)

    response = await asyncio.wait_for(agent._activate(state, body()), timeout=5)

    assert (response.status, response.metadata["mini_swe_termination"]) == ("incomplete", "cancelled")
    invocation, *tools, sandbox_record = state.observations.records
    assert invocation.stop_reason == "cancelled"
    assert (tools[-1].status, tools[-1].error_type) == ("cancelled", "invocation_cancelled")
    assert sandbox_record.outcome == "cancelled"
    assert "native_trajectory_partial" in [gap.code for gap in state.observations.gaps]


async def test_a_submitted_run_that_raced_close_keeps_its_own_outcome(agent) -> None:
    state = state_for(agent)
    output = {"runtime": {"hostname": "box", "pid": 7}, "stop_reason": "cancelled", "error": None, "mcp": None}
    finish_run(state, trajectory=native_trajectory(), output=output)

    response = await asyncio.wait_for(agent._activate(state, body()), timeout=5)

    assert (response.status, response.metadata["mini_swe_termination"]) == ("completed", "completed")
    assert state.observations.records[-1].outcome == "completed"


@pytest.mark.parametrize(
    ("exit_status", "error", "failure_kind", "status_code"),
    [
        ("RateLimitError", "litellm.RateLimitError: 429", "model_request_error", 429),
        ("PaymentRequiredError", "litellm.PaymentRequiredError: 402", "model_request_error", 402),
        ("InternalServerError", "litellm.InternalServerError: 500", "model_request_error", 500),
        ("APIConnectionError", "Connection refused", "model_transport_error", None),
        ("Timeout", "litellm.Timeout: model request timed out", "model_transport_error", None),
        ("RepeatedFormatError", "", "agent_invalid_output", None),
        ("BadRequestError", "This model's maximum context length is 32768 tokens", "model_context_limit", None),
    ],
)
async def test_a_reported_model_failure_is_classified_for_the_environment(
    agent, exit_status, error, failure_kind, status_code
) -> None:
    trajectory = native_trajectory()
    trajectory["info"]["exit_status"] = exit_status
    trajectory["messages"][-1] = {
        "role": "exit",
        "content": error,
        "extra": {"exit_status": exit_status, "exception_str": error},
    }
    state = state_for(agent)
    finish_run(state, trajectory=trajectory, return_code=0 if exit_status == "RepeatedFormatError" else 1)

    response = await agent._activate(state, body())

    if failure_kind == "model_context_limit":
        # A context overflow keeps its partial work and is verified like other limits.
        assert response.status == "incomplete"
        return
    assert response.status == "failed"
    outcome = json.loads(response.metadata[AGENT_OUTCOME_METADATA_KEY])
    assert outcome == {"failure_kind": failure_kind, **({"status_code": status_code} if status_code else {})}
    assert state.observations.records[0].status == "failed"


@pytest.mark.parametrize(
    ("trajectory", "return_code", "output", "match"),
    [
        (None, 1, {"error": {"type": "ImportError", "message": "no litellm"}}, "saved no trajectory"),
        (None, 3, {"error": {"type": "mcp_unavailable", "message": "Required MCP servers did not connect: t"}}, "MCP"),
        ("unknown", 1, {}, "stopped with ValueError"),
    ],
)
async def test_failures_that_are_not_the_agents_raise_for_a_retry(
    agent, trajectory, return_code, output, match
) -> None:
    if trajectory == "unknown":
        trajectory = native_trajectory()
        trajectory["info"]["exit_status"] = "ValueError"
    state = state_for(agent)
    finish_run(state, trajectory=trajectory, return_code=return_code, output=output)

    with pytest.raises(RuntimeError, match=match):
        await agent._activate(state, body())


async def test_unconfirmed_cleanup_is_never_a_gradable_response(agent) -> None:
    state = state_for(agent)
    execute = finish_run(state, trajectory=native_trajectory())
    inner = execute.side_effect

    async def unconfirmed(**kwargs):
        result = await inner(**kwargs)
        state.session.cleanup = {**state.session.cleanup, "cleanup_confirmed": False}
        return result

    execute.side_effect = unconfirmed
    with pytest.raises(RuntimeError, match="without confirmed supervisor cleanup"):
        await agent._activate(state, body())


async def test_mcp_calls_reach_the_response_for_verifiers(agent) -> None:
    state = state_for(agent)
    call = {
        "tool_call_id": "call_1",
        "server": "task_tools",
        "tool": "get_weather",
        "arguments": {"city": "SF"},
        "is_error": False,
        "output": "{}",
    }
    finish_run(state, trajectory=native_trajectory(), mcp_calls=[call])

    response = await agent._activate(state, body())

    assert "mcp__task_tools__get_weather" in [getattr(item, "name", None) for item in response.output]


async def test_mcp_call_records_left_out_are_a_declared_gap(agent) -> None:
    state = state_for(agent)
    finish_run(state, trajectory=native_trajectory(), invalid_mcp_calls=2)

    await asyncio.wait_for(agent._activate(state, body()), timeout=5)

    (gap,) = [gap for gap in state.observations.gaps if gap.code == "mcp_call_record_invalid"]
    assert (gap.invocation_id, gap.detail) == ("rollout", "2")


# -- close -----------------------------------------------------------------------------------------------------------


async def test_close_returns_the_activation_observations_and_cleanup(agent) -> None:
    state = state_for(agent)
    finish_run(state, trajectory=native_trajectory())
    bind(agent, state)
    await asyncio.wait_for(agent.responses(http_request(state), body()), timeout=5)
    state.session.close = AsyncMock()

    closed = await agent._close_agent_session_state(state)

    state.session.close.assert_awaited_once_with(timeout=60)
    assert closed.agent_observations is state.observations
    assert closed.cleanup_confirmed is True


async def test_close_after_a_failed_activation_reports_its_partial_run(agent) -> None:
    state = state_for(agent)
    trajectory = native_trajectory()
    trajectory["info"]["exit_status"] = "ValueError"
    finish_run(state, trajectory=trajectory, return_code=1)
    state.task = asyncio.create_task(agent._activate(state, body()))
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(state.task, timeout=5)
    state.session.close = AsyncMock()

    closed = await agent._close_agent_session_state(state)

    invocation = closed.agent_observations.records[0]
    assert invocation.status == "failed" and len(invocation.model_calls) == 3


async def test_close_without_any_artifacts_reports_the_gap(agent) -> None:
    state = state_for(agent)
    state.session.close = AsyncMock()

    closed = await agent._close_agent_session_state(state)

    assert [gap.code for gap in closed.agent_observations.gaps] == ["observation_capture_failed"]


async def test_close_cancels_a_running_activation_after_stopping_the_harness(agent) -> None:
    state = state_for(agent)
    order = []
    started = asyncio.Event()

    async def run_forever(**kwargs):
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            order.append("activation cancelled")
            raise

    async def close(*, timeout):
        await asyncio.sleep(0.05)  # Stopping the harness and collecting its files take sandbox round trips.
        order.append("harness stopped")

    state.session.execute = AsyncMock(side_effect=run_forever)
    state.session.close = AsyncMock(side_effect=close)
    bind(agent, state)
    waiter = asyncio.create_task(agent.responses(http_request(state), body()))
    await asyncio.wait_for(started.wait(), timeout=5)

    await asyncio.wait_for(agent._close_agent_session_state(state), timeout=5)

    assert order == ["harness stopped", "activation cancelled"]
    assert state.task.cancelled()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiter, timeout=5)


@pytest.mark.parametrize(
    ("timed_out", "stop_reason", "outcome"), [(False, "cancelled", "cancelled"), (True, "wall_time", "timeout")]
)
async def test_close_reports_the_partial_run_of_an_activation_it_stopped(
    agent, timed_out, stop_reason, outcome
) -> None:
    state = state_for(agent)
    trajectory = native_trajectory()
    trajectory["info"]["exit_status"] = ""
    del trajectory["messages"][-1]
    started = asyncio.Event()

    async def run_until_stopped(**kwargs):
        state.session.launch_started = True
        started.set()
        await asyncio.sleep(3600)

    async def stop(*, timeout):
        # The harness was stopped; SandboxSession collected the run's files before release.
        state.session.cleanup = {"cleanup_confirmed": True, "return_code": 143, "timed_out": timed_out, "error": None}
        state.session.artifacts = MiniSWEArtifacts(trajectory=trajectory, output=None, mcp_calls=[], log="")

    state.session.execute = AsyncMock(side_effect=run_until_stopped)
    state.session.close = AsyncMock(side_effect=stop)
    bind(agent, state)
    waiter = asyncio.create_task(agent.responses(http_request(state), body()))
    await asyncio.wait_for(started.wait(), timeout=5)

    closed = await asyncio.wait_for(agent._close_agent_session_state(state), timeout=5)

    invocation, *tools, sandbox_record = closed.agent_observations.records
    assert (invocation.status, invocation.stop_reason) == ("incomplete", stop_reason)
    assert len(invocation.model_calls) == 3
    # The submission command was running when the run stopped: mini-SWE saved no observation for it.
    assert (tools[-1].status, tools[-1].error_type) == (outcome, f"invocation_{outcome}")
    assert (sandbox_record.outcome, sandbox_record.exit_code) == (outcome, 143)
    assert "native_trajectory_partial" in [gap.code for gap in closed.agent_observations.gaps]
    assert closed.cleanup_confirmed is True
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiter, timeout=5)


async def test_an_observation_build_failure_at_close_still_returns_a_gap(agent, monkeypatch) -> None:
    state = state_for(agent)
    state.session.artifacts = MiniSWEArtifacts(trajectory=native_trajectory(), output=None, mcp_calls=[], log="")
    state.session.close = AsyncMock()
    monkeypatch.setattr(MiniSWEAgent, "_observations", MagicMock(side_effect=ValueError("malformed")))

    closed = await asyncio.wait_for(agent._close_agent_session_state(state), timeout=5)

    assert [gap.code for gap in closed.agent_observations.gaps] == ["observation_capture_failed"]


def test_the_session_template_resolves_to_this_agent() -> None:
    from nemo_gym.data.tasks.harbor.prepare import resolve_agent

    selection = resolve_agent("mini_swe_agent/mini_swe_agent_sandboxed")

    assert (selection.instance_name, selection.impl_name, selection.runs_in_sandbox) == (
        "mini_swe_agent",
        "mini_swe_agent",
        True,
    )


def test_the_task_tool_context_capability_is_declared(agent) -> None:
    assert agent.supports_task_tool_context()


def test_rollout_stops_reach_the_supervised_session(agent) -> None:
    state = state_for(agent)

    assert agent._rollout_sandbox_session(state) is state.session


@pytest.mark.parametrize("provider_config", [{"unknown_provider": {}}, {"local": 17}])
@pytest.mark.parametrize("borrowed", [True, False])
def test_invalid_provider_configuration_is_a_final_seed_refusal(
    agent, sandbox, monkeypatch, provider_config, borrowed
):
    from nemo_gym.sandbox import create_provider

    agent.config.sandbox_provider = "sandbox"
    monkeypatch.setattr(app_module, "resolve_provider_config", lambda *args: provider_config)
    monkeypatch.setattr(app_module, "create_provider", create_provider)
    reason = check_agent_seed_rejected(
        agent.setup_webserver(), seed(sandbox_access=SANDBOX_ACCESS if borrowed else None)
    )
    assert "provider" in reason.lower()
    sandbox.connect.assert_not_awaited()
    sandbox.box.start.assert_not_awaited()
    assert not agent._session_records


@pytest.mark.parametrize("typed_cleanup", [True, False])
async def test_connect_failure_survives_cleanup_failure(agent, sandbox, typed_cleanup):
    from nemo_gym.core import NgFailureEvent, RequestFailureLog, failure_report_for

    original = NgFailureEvent(
        boundary="sandbox",
        failure_kind="provider_unavailable",
        failure_reason="connect failed",
        operation="connect",
        stage="seed",
        event_id="original-connect-event",
    )
    cleanup = (
        NgFailureEvent(
            boundary="sandbox",
            failure_kind="session_release_failed",
            failure_reason="cleanup failed",
            operation="aclose",
            stage="cleanup",
            event_id="cleanup-event",
        )
        if typed_cleanup
        else RuntimeError("cleanup failed")
    )
    sandbox.connect.side_effect = original
    sandbox.provider.aclose.side_effect = cleanup
    with RequestFailureLog.capture() as log:
        with pytest.raises(NgFailureEvent) as caught:
            await agent._seed_agent_session_state(seed())
        assert caught.value is original
        report = failure_report_for(caught.value)
        event = report.events[report.failure_id]
        assert (event.event_id, event.stage, event.operation) == ("original-connect-event", "seed", "connect")
        (cleanup_event,) = log.report.events.values()
        assert (cleanup_event.stage, cleanup_event.operation) == ("cleanup", "aclose")
        if typed_cleanup:
            assert cleanup_event.event_id == "cleanup-event"
    sandbox.provider.aclose.assert_awaited_once()


@pytest.mark.parametrize("settings", [{"metadata": 17}, {"unknown_setting": True}])
def test_invalid_owned_sandbox_spec_is_refused_before_provider_allocation(agent, sandbox, monkeypatch, settings):
    agent.config.sandbox_provider = "sandbox"
    agent.config.sandbox_config = {"workdir": "/app", **settings}
    create = MagicMock()
    monkeypatch.setattr(app_module, "create_provider", create)
    assert check_agent_seed_rejected(agent.setup_webserver(), seed(sandbox_access=None))
    create.assert_not_called()
    sandbox.box.start.assert_not_awaited()
    assert not agent._session_records


@pytest.mark.parametrize(
    ("exit_status", "status_code", "failure_kind"),
    [
        # LiteLLM 1.106.0 maps these answers to classes whose names do not say the status.
        ("APIError", 403, "model_request_error"),
        ("APIError", 413, "model_request_error"),
        ("BadRequestError", 422, "model_request_error"),
        ("ContentPolicyViolationError", 400, "model_request_error"),
        # No answer came; a Timeout carries a status attribute of its own.
        ("Timeout", 408, "model_transport_error"),
    ],
)
async def test_a_model_failure_is_typed_from_the_status_of_its_answer(agent, exit_status, status_code, failure_kind):
    trajectory = native_trajectory()
    trajectory["info"]["exit_status"] = exit_status
    trajectory["messages"][-1] = {"role": "exit", "content": "", "extra": {"exit_status": exit_status}}
    state = state_for(agent)
    output = {
        "runtime": {"hostname": "box", "pid": 7, "python": "/py"},
        "error": {"type": exit_status, "message": "refused", "status_code": status_code},
    }
    finish_run(state, trajectory=trajectory, return_code=1, output=output)

    response = await agent._activate(state, body())

    assert response.status == "failed"
    outcome = json.loads(response.metadata[AGENT_OUTCOME_METADATA_KEY])
    expected = {"failure_kind": failure_kind}
    if failure_kind == "model_request_error":
        expected["status_code"] = status_code
    assert outcome == expected


async def test_the_vllm_context_window_reply_ends_the_run_incomplete(agent) -> None:
    trajectory = native_trajectory()
    trajectory["info"]["exit_status"] = "RepeatedFormatError"
    # Gym's vLLM Model Server answered the overflowing prompt with an empty reply that it marked; mini-SWE rejected it.
    rejected = model_response("resp_overflow")
    rejected["output"] = []
    rejected["metadata"] = {"nemo_gym_context_window_exceeded": "true"}
    trajectory["messages"][-1] = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "format error"}],
        "extra": {"interrupt_type": "FormatError", "response": rejected},
    }
    trajectory["messages"].append(
        {"role": "exit", "content": "", "extra": {"exit_status": "RepeatedFormatError", "exception_str": ""}}
    )
    state = state_for(agent)
    finish_run(state, trajectory=trajectory)

    response = await agent._activate(state, body())

    assert response.status == "incomplete"
    assert response.metadata["mini_swe_termination"] == "ContextWindowExceededError"
    assert AGENT_OUTCOME_METADATA_KEY not in response.metadata


@pytest.mark.parametrize("tool_choice", [{"type": "function", "name": "get_weather"}, "none"])
def test_a_tool_choice_the_bash_tool_cannot_serve_is_refused(agent, tool_choice) -> None:
    with pytest.raises(HTTPException) as refused:
        agent._validate_request(body(tool_choice=tool_choice))
    assert refused.value.status_code == 422
    agent._validate_request(body(tool_choice="required"))


async def test_a_model_call_that_saved_no_response_leaves_usage_unknown_with_a_gap(agent) -> None:
    trajectory = {
        "info": {"exit_status": "RateLimitError", "model_stats": {"api_calls": 1}},
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Please solve this issue: fix it"},
            {"role": "exit", "content": "", "extra": {"exit_status": "RateLimitError", "exception_str": "429"}},
        ],
    }
    state = state_for(agent)
    finish_run(state, trajectory=trajectory, return_code=1)

    response = await agent._activate(state, body())

    assert (response.status, response.usage) == ("failed", None)
    assert "usage_unavailable" in [gap.code for gap in state.observations.gaps]


async def test_an_optional_mcp_server_that_did_not_connect_is_a_gap(agent) -> None:
    state = state_for(agent)
    output = {
        "runtime": {"hostname": "box", "pid": 7, "python": "/py"},
        "mcp": {"servers": ["task_tools"], "unavailable": [{"server": "extra", "error": "ConnectError"}]},
    }
    finish_run(state, trajectory=native_trajectory(), output=output)

    await agent._activate(state, body())

    [gap] = [gap for gap in state.observations.gaps if gap.code == "mcp_server_unavailable"]
    assert json.loads(gap.detail) == [{"server": "extra", "error": "ConnectError"}]


async def test_network_access_restricts_a_sandbox_the_agent_starts(agent, sandbox, monkeypatch) -> None:
    agent.config.sandbox_provider = "sandbox"
    agent.config.network_access = "model_and_tools"
    restricted = []

    def restrict(spec, **policy):
        restricted.append(policy)
        return dataclasses.replace(spec, provider_options={"network_policy": "restricted"})

    monkeypatch.setattr(app_module, "restrict_sandbox_network", restrict)

    await agent._seed_agent_session_state(seed(sandbox_access=None, tool_accesses=[mcp_grant(required=True)]))

    [policy] = restricted
    assert policy["network_access"] == "model_and_tools"
    assert policy["model_base_url"].startswith("http://10.0.0.5:8000/")
    assert (policy["tool_urls"], policy["required_tool_urls"]) == (["http://10.0.0.5:9000/mcp"],) * 2
    (spec,), _ = sandbox.box.start.await_args
    assert spec.provider_options == {"network_policy": "restricted"}


async def test_network_access_cannot_restrict_a_borrowed_sandbox(agent, sandbox) -> None:
    agent.config.network_access = "model_only"

    with pytest.raises(AgentSeedRejectedError, match="only to a sandbox it starts"):
        await agent._seed_agent_session_state(seed())
    sandbox.connect.assert_not_awaited()


async def test_a_network_policy_the_provider_cannot_enforce_closes_the_provider(agent, sandbox, monkeypatch) -> None:
    agent.config.sandbox_provider = "sandbox"
    agent.config.network_access = "model_only"
    monkeypatch.setattr(
        app_module, "restrict_sandbox_network", MagicMock(side_effect=AgentSeedRejectedError("no network policies"))
    )

    with pytest.raises(AgentSeedRejectedError, match="no network policies"):
        await agent._seed_agent_session_state(seed(sandbox_access=None))
    sandbox.provider.aclose.assert_awaited_once()
    sandbox.box.start.assert_not_awaited()
