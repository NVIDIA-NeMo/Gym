# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent sessions: seed, one activation, close, and how a mini-SWE run becomes a response."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    SandboxSession,
    check_agent_seed_rejected,
)
from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, NeMoGymResponseCreateParamsNonStreaming, TaskId
from nemo_gym.server_runtime import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess
from responses_api_agents.mini_swe_agent import app as app_module
from responses_api_agents.mini_swe_agent.app import MiniSWEAgent, MiniSWEAgentConfig
from responses_api_agents.mini_swe_agent.sandbox import HarnessProcessInfo, MiniSWEArtifacts, MiniSWESession
from responses_api_agents.mini_swe_agent.tests.test_trajectory import native_trajectory


SANDBOX_ACCESS = {
    "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {"sandbox_id": "task-box"}},
    "workdir": "/app",
}


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
) -> AsyncMock:
    """Make the session's execute end as a run that left these artifacts and this supervisor receipt."""
    artifacts = MiniSWEArtifacts(
        trajectory=trajectory,
        output=output if output is not None else {"runtime": {"hostname": "box", "pid": 7, "python": "/py"}},
        mcp_calls=mcp_calls or [],
        log="",
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
        ({"sandbox_access": SANDBOX_ACCESS | {"workdir": "/tmp"}}, "outside its session directory"),
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

    first, second = await asyncio.gather(
        agent.responses(http_request(state), body()), agent.responses(http_request(state), body())
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


def test_the_payload_routes_every_model_call_through_the_rollout_and_forwards_sampling(agent) -> None:
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
        max_output_tokens=8,
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
    assert kwargs["extra_body"] == {"max_output_tokens": 8}
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


async def test_a_run_killed_mid_step_is_the_agents_own_result(agent) -> None:
    trajectory = native_trajectory()
    trajectory["info"]["exit_status"] = ""
    state = state_for(agent)
    finish_run(state, trajectory=trajectory, return_code=137)

    response = await agent._activate(state, body())

    assert (response.status, response.metadata["mini_swe_termination"]) == ("incomplete", "killed")


@pytest.mark.parametrize(
    ("exit_status", "error", "failure_kind", "status_code"),
    [
        ("RateLimitError", "litellm.RateLimitError: 429", "model_request_error", 429),
        ("InternalServerError", "litellm.InternalServerError: 500", "model_request_error", 500),
        ("APIConnectionError", "Connection refused", "model_transport_error", None),
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


# -- close -----------------------------------------------------------------------------------------------------------


async def test_close_returns_the_activation_observations_and_cleanup(agent) -> None:
    state = state_for(agent)
    finish_run(state, trajectory=native_trajectory())
    bind(agent, state)
    await agent.responses(http_request(state), body())
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
        await state.task
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
        await asyncio.sleep(3600)

    async def close(*, timeout):
        order.append("harness stopped")

    state.session.execute = AsyncMock(side_effect=run_forever)
    state.session.close = AsyncMock(side_effect=close)
    bind(agent, state)
    waiter = asyncio.create_task(agent.responses(http_request(state), body()))
    await asyncio.wait_for(started.wait(), timeout=5)

    await asyncio.wait_for(agent._close_agent_session_state(state), timeout=5)

    assert order == ["harness stopped"]
    assert state.task.cancelled()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiter, timeout=5)


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
