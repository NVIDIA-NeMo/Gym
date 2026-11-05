# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent sessions of the sandboxed DeepAgents agent, on the local sandbox provider in a temporary directory.

The episode tests run the real sandbox runner under Gym's process supervisor, which needs Linux.
"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Response
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.agents import (
    ROLLOUT_STOP_METADATA_KEY,
    AgentCloseSessionRequest,
    AgentSeedSessionRequest,
    SandboxPythonRuntime,
    check_agent_seed_rejected,
)
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
    EpisodeId,
    NeMoGymResponseCreateParamsNonStreaming,
    TaskId,
)
from nemo_gym.sandbox import (
    AsyncSandbox,
    DirectSandboxConnection,
    SandboxAccess,
    SandboxSpec,
    get_provider_class,
    register_provider,
)
from nemo_gym.server_runtime import DirectHTTPToolAccess
from responses_api_agents.langchain_deepagents_agent import app, sandbox
from responses_api_agents.langchain_deepagents_agent.app import (
    DeepAgentsAgent,
    DeepAgentsAgentConfig,
    _usage,
)
from responses_api_agents.langchain_deepagents_agent.tests.fakes import USAGE, ModelServer, call, text


linux_only = pytest.mark.skipif(sys.platform != "linux", reason="Gym's process supervisor needs Linux")
_EPISODE = EpisodeId(rollout_id="rollout")


def _agent(**overrides) -> DeepAgentsAgent:
    config = DeepAgentsAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="langchain_deepagents_sandbox_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        max_input_tokens=100_000,
        **overrides,
    )
    return DeepAgentsAgent(config=config, server_client=MagicMock(spec=ServerClient))


def _seed(**fields) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="agent-session", episode_id=_EPISODE, task_id=TaskId(taskset="tb", task_id="t"), **fields
    )


def _access(workdir: str = "/app", **fields) -> SandboxAccess:
    return SandboxAccess(
        connection=DirectSandboxConnection(provider_config_ref="lender", descriptor={"sandbox_id": "sb"}),
        workdir=workdir,
        **fields,
    )


def _request(rollout_id: str = "rollout") -> SimpleNamespace:
    return SimpleNamespace(session={}, cookies={}, path_params={"rollout_id": rollout_id})


def _close_body() -> AgentCloseSessionRequest:
    return AgentCloseSessionRequest(agent_session_id="agent-session", episode_id=_EPISODE)


def _sandbox_record(close) -> SimpleNamespace:
    [record] = [record for record in close.agent_observations.records if record.kind == "sandbox"]
    return record


# --- configuration, seed refusals and request mapping -----------------------------------------------------


@pytest.mark.parametrize(
    ("seed", "reason"),
    [
        (
            _seed(
                sandbox_access=_access(),
                tool_accesses=[DirectHTTPToolAccess(name="task", required=True, base_url="http://r", cookies={})],
            ),
            "Sandboxed DeepAgents does not support required tool access: task",
        ),
        (_seed(), "Sandboxed DeepAgents requires sandbox_access or a sandbox_provider"),
        (_seed(sandbox_access=_access(user="agent")), "Sandboxed DeepAgents does not support a sandbox user override"),
        (_seed(sandbox_access=_access("app")), "Sandboxed DeepAgents requires an absolute sandbox workdir"),
    ],
)
def test_a_seed_the_agent_cannot_serve_is_refused(seed, reason):
    assert check_agent_seed_rejected(_agent().setup_webserver(), seed) == reason


def test_a_loopback_model_url_is_refused_for_a_remote_sandbox(monkeypatch):
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {"lender": {"opensandbox": {}}})
    monkeypatch.setattr(
        DeepAgentsAgent, "resolve_model_base_url", lambda self, *args, **kwargs: "http://127.0.0.1:8000/v1"
    )

    reason = check_agent_seed_rejected(_agent().setup_webserver(), _seed(sandbox_access=_access()))

    assert "loopback address" in reason


def test_a_started_sandbox_needs_an_absolute_workdir():
    with pytest.raises(ValidationError, match="sandbox_config.workdir must be an absolute path"):
        _agent(sandbox_provider="sandbox", sandbox_config={})


def test_the_agent_runs_only_in_agent_sessions():
    client = TestClient(_agent().setup_webserver())

    assert client.post("/v1/responses", json={"input": "hi"}).status_code == 409
    assert client.post("/run", json={"responses_create_params": {"input": "hi"}}).status_code == 409


def test_usage_is_known_only_when_every_model_call_returned_it():
    assert _usage({"model_calls": 2, "usages": [USAGE, USAGE]}).total_tokens == 30
    assert _usage({"model_calls": 2, "usages": [USAGE, None]}) is None
    assert _usage({"model_calls": 2, "usages": [USAGE]}) is None


def test_a_model_failure_reports_its_cause_to_the_environment():
    agent = _agent()

    request_error = agent.classify_outcome({"model_error": "HTTP 429", "model_error_status": 429})
    transport_error = agent.classify_outcome({"model_error": "refused", "model_transport_error": True})

    assert (request_error.failure_kind, request_error.status_code) == (MODEL_REQUEST_ERROR, 429)
    assert transport_error.failure_kind == MODEL_TRANSPORT_ERROR


# --- episodes on the local provider -----------------------------------------------------------------------


class _LendableLocalProvider(get_provider_class("local")):
    """The local provider, made connectable by sandbox id so a test can lend its sandbox to the agent, as a
    Resources Server lends a Docker or OpenSandbox sandbox. Records each provider close of a sandbox."""

    name = "lendable_local"
    handles: ClassVar[dict] = {}
    closed: ClassVar[list] = []

    async def serialize_handle(self, handle, *, scope=None) -> dict:
        self.handles[handle.sandbox_id] = handle
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor):
        return self.handles[descriptor["sandbox_id"]]

    async def close(self, handle) -> None:
        self.closed.append(handle.sandbox_id)
        await super().close(handle)


@pytest.fixture
def runtime(tmp_path, monkeypatch) -> Path:
    """The host's deepagents install stands in for the sandbox runtime; the harness files go under tmp_path."""
    package = tmp_path / "runtime/harness/responses_api_agents/langchain_deepagents_agent"
    monkeypatch.setattr(SandboxPythonRuntime, "is_installed", AsyncMock(return_value=True))
    monkeypatch.setattr(sandbox, "SANDBOX_PYTHON", sys.executable)
    monkeypatch.setattr(sandbox, "_PACKAGE_DIR", str(package))
    return package


@pytest.fixture
async def lent_sandbox(tmp_path, monkeypatch):
    """A running task sandbox owned by the test (the Resources Server's role), with a task file in its workdir."""
    register_provider("lendable_local", _LendableLocalProvider, override=True)
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {"lender": {"lendable_local": {}}})
    # The lendable provider shares the host's network as the local one does.
    monkeypatch.setattr(app, "assert_model_url_reachable_from_sandbox", lambda url, *, provider_name: None)
    # Episodes that call the model point this at their stub Model Server.
    monkeypatch.setattr(DeepAgentsAgent, "resolve_model_base_url", lambda self, *args, **kwargs: "http://model/v1")
    workdir = tmp_path / "task"
    workdir.mkdir()
    (workdir / "input.txt").write_text("from the task image\n")
    owner = AsyncSandbox(_LendableLocalProvider())
    await owner.start(SandboxSpec(workdir=str(workdir)))
    access = SandboxAccess(
        connection=DirectSandboxConnection(provider_config_ref="lender", descriptor=await owner.serialize()),
        workdir=str(workdir),
        env={"TASK_VAR": "from-the-owner"},
    )
    yield owner, access
    await owner.stop()


def _model(monkeypatch, answers) -> ModelServer:
    model = ModelServer(answers)
    monkeypatch.setattr(DeepAgentsAgent, "resolve_model_base_url", lambda self, *args, **kwargs: model.base_url)
    return model


async def test_a_failed_runtime_install_releases_the_lent_sandbox(lent_sandbox, runtime, monkeypatch):
    owner, access = lent_sandbox
    monkeypatch.setattr(SandboxPythonRuntime, "is_installed", AsyncMock(return_value=False))
    monkeypatch.setattr(SandboxPythonRuntime, "install", AsyncMock(side_effect=RuntimeError("no wheel")))

    with pytest.raises(RuntimeError, match="no wheel"):
        await _agent().seed_agent_session(_request(), _seed(sandbox_access=access))

    assert owner.sandbox_id not in _LendableLocalProvider.closed
    assert (await owner.exec("cat input.txt")).stdout == "from the task image\n"


@linux_only
async def test_an_episode_runs_the_graph_in_the_lent_sandbox_and_close_leaves_it_running(
    lent_sandbox, runtime, monkeypatch
):
    owner, access = lent_sandbox
    workdir = Path(access.workdir)
    model = _model(
        monkeypatch,
        [[call("execute", {"command": "echo $TASK_VAR; cp input.txt copied.txt"}, "c1")], [text("done")]],
    )
    agent, request = _agent(), _request()
    try:
        await agent.seed_agent_session(request, _seed(sandbox_access=access))
        body = NeMoGymResponseCreateParamsNonStreaming(input="Do the task.", temperature=0.5)
        response = await asyncio.wait_for(agent.responses(request, Response(), body), 120)
        close = await asyncio.wait_for(agent.close_agent_session(request, _close_body()), 60)
    finally:
        model.close()

    assert response.status == "completed"
    assert [item.type for item in response.output] == ["function_call", "function_call_output", "message"]
    assert response.output[1].output.startswith("from-the-owner\n")
    assert response.output[2].content[0].text == "done"
    assert response.usage.total_tokens == 30
    assert {request["temperature"] for request in model.requests} == {0.5}
    assert (workdir / "copied.txt").read_text() == "from the task image\n"
    # The harness's runtime and session files stay outside the task's working directory.
    assert sorted(path.name for path in workdir.iterdir()) == ["copied.txt", "input.txt"]
    record = _sandbox_record(close)
    assert (record.role, record.sandbox_id, record.outcome) == ("agent", owner.sandbox_id, "completed")
    assert [gap.code for gap in close.agent_observations.gaps] == [
        "agent_transcript_unavailable",
        "model_call_ownership_unavailable",
    ]
    # Close disconnected without stopping: the owner still runs commands in it and stops it itself.
    assert owner.sandbox_id not in _LendableLocalProvider.closed
    assert (await owner.exec("cat copied.txt")).stdout == "from the task image\n"


@linux_only
async def test_a_run_past_the_task_timeout_answers_with_its_partial_run_as_incomplete(
    lent_sandbox, runtime, monkeypatch
):
    _, access = lent_sandbox
    model = _model(monkeypatch, [[call("execute", {"command": "echo first"}, "c1")], None])
    agent, request = _agent(), _request()
    try:
        await agent.seed_agent_session(request, _seed(sandbox_access=access))
        body = NeMoGymResponseCreateParamsNonStreaming(input="Do the task.", metadata={"agent_timeout_sec": "5"})
        response = await asyncio.wait_for(agent.responses(request, Response(), body), 120)
        close = await asyncio.wait_for(agent.close_agent_session(request, _close_body()), 60)
    finally:
        model.close()

    assert (response.status, response.metadata["stop_reason"]) == ("incomplete", "wall_time")
    assert response.output[1].output.startswith("first")
    # The second model call never returned, so the run's usage is unknown.
    assert response.usage is None
    assert _sandbox_record(close).outcome == "timeout"


@linux_only
async def test_close_stops_a_running_graph_before_releasing_the_lent_sandbox(lent_sandbox, runtime, monkeypatch):
    owner, access = lent_sandbox
    model = _model(monkeypatch, [None])
    agent, request = _agent(), _request()
    try:
        await agent.seed_agent_session(request, _seed(sandbox_access=access))
        activation = asyncio.create_task(
            agent.responses(request, Response(), NeMoGymResponseCreateParamsNonStreaming(input="Do the task."))
        )
        assert await asyncio.to_thread(model.blocked.wait, 60), "the graph never called the model"
        close = await asyncio.wait_for(agent.close_agent_session(request, _close_body()), 60)
    finally:
        model.close()

    assert _sandbox_record(close).outcome == "cancelled"
    # The stopped runner reports its partial run, which can end the activation before close cancels it.
    [ending] = await asyncio.wait_for(asyncio.gather(activation, return_exceptions=True), 10)
    if not isinstance(ending, asyncio.CancelledError):
        assert (ending.status, ending.metadata["stop_reason"]) == ("incomplete", "cancelled")
    assert owner.sandbox_id not in _LendableLocalProvider.closed


@linux_only
async def test_a_rollout_stop_ends_the_run_as_a_marked_timeout(lent_sandbox, runtime, monkeypatch):
    _, access = lent_sandbox
    model = _model(monkeypatch, [None])
    agent, request = _agent(), _request()
    try:
        await agent.seed_agent_session(request, _seed(sandbox_access=access))
        activation = asyncio.create_task(
            agent.responses(request, Response(), NeMoGymResponseCreateParamsNonStreaming(input="Do the task."))
        )
        assert await asyncio.to_thread(model.blocked.wait, 60), "the graph never called the model"
        stop = await agent.stop_rollout(_EPISODE.capture_key)
        response = await asyncio.wait_for(activation, 60)
        await asyncio.wait_for(agent.close_agent_session(request, _close_body()), 60)
    finally:
        model.close()

    assert stop.stopped
    assert response.status == "incomplete"
    assert json.loads(response.metadata[ROLLOUT_STOP_METADATA_KEY]) == {"reason": None}


@linux_only
async def test_a_started_sandbox_is_stopped_at_close(tmp_path, runtime, monkeypatch):
    workdir = tmp_path / "owned"
    workdir.mkdir()
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {"sandbox": {"local": {}}})
    model = _model(monkeypatch, [[text("done")]])
    agent = _agent(sandbox_provider="sandbox", sandbox_config={"workdir": str(workdir)})
    try:
        state = await agent._seed_agent_session_state(_seed())
        response = await asyncio.wait_for(
            agent._run_in_sandbox(NeMoGymResponseCreateParamsNonStreaming(input="Do the task."), state), 120
        )
        await asyncio.wait_for(agent._close_agent_session_state(state), 60)
    finally:
        model.close()

    assert response.status == "completed"
    assert state.session.owns_sandbox and state.session.sandbox_stopped


async def test_a_request_with_its_own_tools_is_refused(lent_sandbox, runtime):
    _, access = lent_sandbox
    agent = _agent()
    state = await agent._seed_agent_session_state(_seed(sandbox_access=access))
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="Do the task.", tools=[{"type": "function", "name": "lookup", "parameters": {}, "strict": False}]
    )
    try:
        with pytest.raises(Exception, match="declares tools") as refused:
            await agent._run_in_sandbox(body, state)
    finally:
        await agent._close_agent_session_state(state)

    assert refused.value.status_code == 422
