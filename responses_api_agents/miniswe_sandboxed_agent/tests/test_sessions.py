# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode sessions: the environment server seeds a borrowed sandbox and calls /v1/responses."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId
from responses_api_agents.miniswe_sandboxed_agent import app as module
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessOutcome


TASK_CONTEXT = {"mcp_servers": [{"name": "fs", "url": "http://localhost:7000/mcp"}], "skills_dir": "/skills"}
CLOSE_BODY = {"agent_session_id": "ag-1", "episode_id": {"rollout_id": "r1", "attempt": 0}}


def exec_result(*, stdout="", stderr="", return_code=0, error_type=None):
    """What ``sandbox.exec`` returns: a provider failure carries ``error_type`` and the 125 sentinel."""
    return SimpleNamespace(stdout=stdout, stderr=stderr, return_code=return_code, error_type=error_type)


class FakeHarness:
    """Records how the agent built it and returns a canned run."""

    instances: list["FakeHarness"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.budget = None
        self.closed = False
        FakeHarness.instances.append(self)

    async def setup(self) -> None:
        pass

    async def execute(self, budget):
        self.budget = budget
        extra = {"ng_agent_observations": {"source": "miniswe", "records": []}}
        return response_stub(), HarnessOutcome(reason="completed"), extra

    async def close(self) -> None:
        self.closed = True


def make_agent(tmp_path, monkeypatch, *, task_context=TASK_CONTEXT):
    FakeHarness.instances.clear()
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    agent = module.MiniSWESandboxedAgent(
        config=module.MiniSWESandboxedConfig(
            host="localhost",
            port=1,
            name="miniswe_sandboxed_agent",
            entrypoint="app.py",
            model_server={"type": "responses_api_models", "name": "policy_model"},
            artifacts_dir=tmp_path,
            agent_timeout_sec=120,
        ),
        server_client=server_client,
    )
    provider = SimpleNamespace(aclose=AsyncMock())

    async def exec_(command, **_kwargs):
        if command.startswith(f"cat {module.TASK_CONTEXT_FILE}"):
            return exec_result(stdout=json.dumps(task_context) if task_context else "")
        return exec_result(stdout="/app\n")

    sandbox = SimpleNamespace(exec=AsyncMock(side_effect=exec_), disconnect=AsyncMock())
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}})
    monkeypatch.setattr(module, "resolve_provider_config", lambda ref, config: {"opensandbox": {}})
    monkeypatch.setattr(module, "create_provider", MagicMock(return_value=provider))
    monkeypatch.setattr(module, "get_server_url", lambda name: "http://policy:1")
    monkeypatch.setattr(module, "MiniSWEHarness", FakeHarness)
    monkeypatch.setattr(type(agent), "base_url_for_run", lambda self, *, base_url, body: base_url)
    connect = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(module.AsyncSandbox, "connect", connect)
    return agent, TestClient(agent.setup_webserver()), sandbox, provider, connect


def seed_body(*, session="ag-1", rollout="r1", with_sandbox=True, workdir="/app"):
    body = {
        "agent_session_id": session,
        "episode_id": {"rollout_id": rollout, "attempt": 0},
        "task_id": {"taskset": "tb4", "task_id": "atrx-vep-crispr"},
        "tool_accesses": [],
    }
    if with_sandbox:
        body["sandbox_access"] = {
            "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {"sandbox_id": "sb"}},
            "workdir": workdir,
        }
    return body


def response_stub(text="done"):
    return module.NeMoGymResponse(
        id="resp",
        created_at=0,
        model="policy_model",
        object="response",
        output=[
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        tool_choice="auto",
        tools=[],
        parallel_tool_calls=False,
    )


def capture_path(rollout="r1"):
    return f"/ng-rollout/{EpisodeId(rollout_id=rollout, attempt=0).capture_key}/v1/responses"


def test_seed_connects_and_the_turn_runs_the_harness_on_that_sandbox(tmp_path, monkeypatch):
    agent, client, sandbox, provider, connect = make_agent(tmp_path, monkeypatch)

    seeded = client.post("/v1/agent_sessions", json=seed_body())
    assert seeded.status_code == 200, seeded.text
    # The environment server lets a task with MCP servers or skills through to this harness, which reads them.
    assert seeded.json() == {"agent_session_id": "ag-1", "supports_task_tool_context": True}
    connect.assert_awaited_once()
    assert connect.await_args.args[0] == {"sandbox_id": "sb"}

    response = client.post(capture_path(), json={"input": [{"role": "user", "content": "Write the report."}]})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["output"][0]["content"][0]["text"] == "done"
    assert payload["metadata"]["miniswe_termination"] == "completed"
    assert payload["metadata"]["miniswe_agent_started"] == "true"

    (harness,) = FakeHarness.instances
    assert harness.kwargs["sandbox"] is sandbox
    context = harness.kwargs["context"]
    assert (context.session_id, context.task_id, context.instruction, context.workdir) == (
        "ag-1",
        "atrx-vep-crispr",
        "Write the report.",
        "/app",
    )
    # The instruction is the turn's input; the tool grants come from the resources server's task.json.
    assert context.mcp_servers == TASK_CONTEXT["mcp_servers"] and context.skills_dir == "/skills"
    assert context.rollout_id == EpisodeId(rollout_id="r1", attempt=0).capture_key
    assert harness.kwargs["model_base_url"] == "http://policy:1/v1"
    assert harness.kwargs["directory"] == tmp_path / "ag-1"
    assert 0 < harness.budget <= 120
    # The harness is not closed after a started run; the sandbox outlives the agent until close.
    assert not harness.closed
    # `pwd` is never needed when the resources server hands over the working directory.
    assert all(not call.args[0].strip() == "pwd" for call in sandbox.exec.await_args_list)

    close = client.post(
        "/v1/agent_sessions/close",
        json={"agent_session_id": "ag-1", "episode_id": {"rollout_id": "r1", "attempt": 0}},
    )
    assert close.status_code == 200, close.text
    assert close.json()["agent_observations"]["source"] == "miniswe"
    sandbox.disconnect.assert_awaited_once()
    provider.aclose.assert_awaited_once()


def test_plain_responses_path_binds_the_only_session_and_disables_capture(tmp_path, monkeypatch):
    agent, client, _, _, _ = make_agent(tmp_path, monkeypatch, task_context=None)
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200
    response = client.post("/v1/responses", json={"input": "Do it"})
    assert response.status_code == 200, response.text
    (harness,) = FakeHarness.instances
    assert harness.kwargs["context"].instruction == "Do it"
    assert harness.kwargs["context"].mcp_servers == [] and harness.kwargs["context"].skills_dir is None
    key = next(iter(agent._sessions))
    assert agent._sessions[key].capture_model_calls is False


def test_a_second_different_turn_on_the_same_session_is_refused(tmp_path, monkeypatch):
    _, client, _, _, _ = make_agent(tmp_path, monkeypatch)
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200
    assert client.post(capture_path(), json={"input": "first"}).status_code == 200
    assert client.post(capture_path(), json={"input": "second"}).status_code == 409


def test_session_identity_rules(tmp_path, monkeypatch):
    _, client, sandbox, _, connect = make_agent(tmp_path, monkeypatch)
    assert client.post("/v1/responses", json={"input": "x"}).status_code == 409
    assert client.post("/v1/agent_sessions", json=seed_body(with_sandbox=False)).status_code == 422
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200  # idempotent
    assert connect.await_count == 1
    assert client.post("/v1/agent_sessions", json=seed_body(rollout="other")).status_code == 409
    wrong = client.post(
        "/v1/agent_sessions/close",
        json={"agent_session_id": "ag-1", "episode_id": {"rollout_id": "other", "attempt": 0}},
    )
    assert wrong.status_code == 409 and not sandbox.disconnect.await_count
    right = client.post(
        "/v1/agent_sessions/close",
        json={"agent_session_id": "ag-1", "episode_id": {"rollout_id": "r1", "attempt": 0}},
    )
    assert right.status_code == 200
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 409


def test_run_without_a_resources_server_is_refused(tmp_path, monkeypatch):
    _, client, _, _, _ = make_agent(tmp_path, monkeypatch)
    response = client.post("/run", json={"responses_create_params": {"input": []}, "rollout_id": "r"})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "value,expected",
    [
        ("plain", "plain"),
        (
            [{"role": "user", "content": "a"}, {"role": "user", "content": [{"type": "input_text", "text": "b"}]}],
            "a\n\nb",
        ),
    ],
)
def test_instruction_from_input(value, expected):
    assert module._instruction(value) == expected


def test_loopback_model_url_on_a_remote_provider_is_refused_at_seed(tmp_path, monkeypatch):
    _, client, _, provider, connect = make_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "get_server_url", lambda name: "http://127.0.0.1:1")

    response = client.post("/v1/agent_sessions", json=seed_body())

    assert response.status_code == 422, response.text
    assert "loopback" in response.json()["detail"]
    connect.assert_not_awaited()
    provider.aclose.assert_not_awaited()


def test_the_task_timeout_bounds_the_turn_budget(tmp_path, monkeypatch):
    _, client, _, _, _ = make_agent(tmp_path, monkeypatch)
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    body = {"input": "Do it", "metadata": {module.AGENT_TIMEOUT_METADATA_KEY: "30"}}
    assert client.post(capture_path(), json=body).status_code == 200
    (harness,) = FakeHarness.instances
    # The configured 120 s ceiling is bounded by the task's own [agent].timeout_sec.
    assert 0 < harness.budget <= 30


@pytest.mark.parametrize("reason", ["infrastructure_error", "cancelled"])
def test_infrastructure_outcomes_are_a_503_so_the_episode_is_masked(tmp_path, monkeypatch, reason):
    class FailingHarness(FakeHarness):
        async def execute(self, budget):
            self.budget = budget
            return response_stub(), HarnessOutcome(reason=reason, detail="sandbox exec lost"), {}

    _, client, sandbox, provider, _ = make_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "MiniSWEHarness", FailingHarness)
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    response = client.post(capture_path(), json={"input": "Do it"})
    assert response.status_code == 503, response.text
    assert reason in response.json()["detail"] and "sandbox exec lost" in response.json()["detail"]
    # A retry of the same turn replays the refusal without running the harness again.
    assert client.post(capture_path(), json={"input": "Do it"}).status_code == 503
    assert len(FakeHarness.instances) == 1

    close = client.post(
        "/v1/agent_sessions/close",
        json={"agent_session_id": "ag-1", "episode_id": {"rollout_id": "r1", "attempt": 0}},
    )
    assert close.status_code == 200, close.text
    sandbox.disconnect.assert_awaited_once()
    provider.aclose.assert_awaited_once()


def test_agent_outcomes_are_verified_not_masked(tmp_path, monkeypatch):
    class TimedOutHarness(FakeHarness):
        async def execute(self, budget):
            self.budget = budget
            return response_stub(), HarnessOutcome(reason="nonzero_exit", detail="LimitsExceeded"), {}

    _, client, _, _, _ = make_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "MiniSWEHarness", TimedOutHarness)
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    response = client.post(capture_path(), json={"input": "Do it"})
    assert response.status_code == 200, response.text
    assert response.json()["metadata"]["miniswe_termination"] == "nonzero_exit"


def test_a_setup_timeout_is_a_503_not_the_agents_answer(tmp_path, monkeypatch):
    class SlowSetup(FakeHarness):
        async def setup(self) -> None:
            await asyncio.sleep(5)

    agent, client, _, _, _ = make_agent(tmp_path, monkeypatch)
    agent.config.setup_timeout_sec = 0.05
    monkeypatch.setattr(module, "MiniSWEHarness", SlowSetup)
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    response = client.post(capture_path(), json={"input": "Do it"})

    # The harness never started, so the untouched sandbox must not be graded as the agent's answer.
    assert response.status_code == 503, response.text
    assert "setup timeout" in response.json()["detail"]
    (harness,) = FakeHarness.instances
    assert harness.closed and harness.budget is None
    assert client.post(capture_path(), json={"input": "Do it"}).status_code == 503


async def test_overlapping_first_turns_bind_once_and_only_when_complete(tmp_path, monkeypatch):
    agent, _, sandbox, _, _ = make_agent(tmp_path, monkeypatch)
    gate, reads = asyncio.Event(), []

    async def blocked_read(sandbox):
        reads.append(sandbox)
        await gate.wait()
        return TASK_CONTEXT

    monkeypatch.setattr(module, "_read_task_context", blocked_read)
    state = module.MiniSWESession(
        sandbox=sandbox,
        seed=module.SeedSessionResponse(session_id="ag-1", task_id="t", agent_timeout_sec=120),
        original_params=None,
        rollout_id="r1",
        capture_model_calls=True,
    )
    body = module.NeMoGymResponseCreateParamsNonStreaming(input="Do it")
    first = asyncio.create_task(agent._bind_episode_turn(state, body, "rk"))
    await asyncio.sleep(0)
    # Nothing is published while the task context is still being read: a second request still sees an
    # unbound state, binds too, and waits behind the first instead of starting with empty grants.
    assert reads == [sandbox] and state.original_params is None
    second = asyncio.create_task(agent._bind_episode_turn(state, body.model_copy(deep=True), "rk"))
    await asyncio.sleep(0)
    assert not second.done()
    gate.set()
    await asyncio.gather(first, second)
    assert len(reads) == 1
    assert state.original_params == body and state.seed.instruction == "Do it"
    assert state.seed.mcp_servers == TASK_CONTEXT["mcp_servers"] and state.seed.skills_dir == "/skills"
    assert state.rollout_id == "rk" and state.capture_model_calls


async def test_a_failed_task_context_read_leaves_the_turn_unbound(tmp_path, monkeypatch):
    agent, _, sandbox, _, _ = make_agent(tmp_path, monkeypatch)

    async def lost(sandbox):
        raise RuntimeError("sandbox exec lost")

    monkeypatch.setattr(module, "_read_task_context", lost)
    state = module.MiniSWESession(
        sandbox=sandbox,
        seed=module.SeedSessionResponse(session_id="ag-1", task_id="t", agent_timeout_sec=120),
        original_params=None,
        rollout_id="r1",
        capture_model_calls=True,
    )
    with pytest.raises(RuntimeError, match="sandbox exec lost"):
        await agent._bind_episode_turn(state, module.NeMoGymResponseCreateParamsNonStreaming(input="Do it"), "rk")
    assert state.original_params is None and state.seed.instruction == "" and state.rollout_id == "r1"


@pytest.mark.parametrize("failure", [{"error_type": "timeout"}, {"error_type": "sandbox"}, {}])
def test_a_provider_failure_reading_the_task_context_is_not_an_absent_file(tmp_path, monkeypatch, failure):
    """A read the provider could not run (exit 125 with an ``error_type``, or any other non-zero exit) must not
    bind the turn without the task's declared tools: the turn is refused as infrastructure, stays unbound, and the
    retry reads again."""
    agent, client, sandbox, _, _ = make_agent(tmp_path, monkeypatch)
    reads = []

    async def exec_(command, **_kwargs):
        if command.startswith(f"cat {module.TASK_CONTEXT_FILE}"):
            reads.append(command)
            if len(reads) == 1:
                return exec_result(stderr="exec did not run", return_code=125, **failure)
            return exec_result(stdout=json.dumps(TASK_CONTEXT))
        return exec_result(stdout="/app\n")

    sandbox.exec.side_effect = exec_
    assert client.post("/v1/agent_sessions", json=seed_body()).status_code == 200

    first = client.post(capture_path(), json={"input": "Do it"})

    assert first.status_code == 503, first.text
    assert "infrastructure_error" in first.json()["detail"] and module.TASK_CONTEXT_FILE in first.json()["detail"]
    key = next(iter(agent._sessions))
    assert agent._sessions[key].original_params is None and agent._sessions[key].worker is None
    assert FakeHarness.instances == []

    retry = client.post(capture_path(), json={"input": "Do it"})

    assert retry.status_code == 200, retry.text
    assert len(reads) == 2
    (harness,) = FakeHarness.instances
    assert harness.kwargs["context"].mcp_servers == TASK_CONTEXT["mcp_servers"]
    assert harness.kwargs["context"].skills_dir == "/skills"


async def test_close_during_a_running_turn_stops_the_harness_before_disconnecting(tmp_path, monkeypatch):
    """The environment server gives up on a turn that is still executing: close cancels and drains it while the
    sandbox is connected, keeps the observations the harness produced on the way out, and only then disconnects.
    The abandoned turn itself answers as a masked cancellation."""
    agent, _, sandbox, provider, _ = make_agent(tmp_path, monkeypatch)
    started, events = asyncio.Event(), []

    class BlockedHarness(FakeHarness):
        async def execute(self, budget):
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                # The harness stops its process groups over the connection: it must still be open here.
                events.append(("harness cancelled", sandbox.disconnect.await_count))
                extra = {"ng_agent_observations": {"source": "miniswe", "records": []}}
                return response_stub(), HarnessOutcome(reason="cancelled"), extra
            raise AssertionError("the turn was not cancelled")

    async def disconnect():
        events.append("disconnect")

    monkeypatch.setattr(module, "MiniSWEHarness", BlockedHarness)
    sandbox.disconnect = AsyncMock(side_effect=disconnect)
    transport = ASGITransport(app=agent.setup_webserver())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.post("/v1/agent_sessions", json=seed_body())).status_code == 200
        turn = asyncio.create_task(client.post(capture_path(), json={"input": "Do it"}))
        await asyncio.wait_for(started.wait(), 5)

        close = await client.post("/v1/agent_sessions/close", json=CLOSE_BODY)

        assert close.status_code == 200, close.text
        assert close.json()["agent_observations"]["source"] == "miniswe"
        assert events == [("harness cancelled", 0), "disconnect"]
        response = await turn
        assert response.status_code == 503 and "cancelled" in response.json()["detail"]
        assert agent._sessions == {}
        # A retried close replays the receipt, with the observations, without touching the sandbox again.
        again = await client.post("/v1/agent_sessions/close", json=CLOSE_BODY)
        assert again.status_code == 200 and again.json()["agent_observations"]["source"] == "miniswe"
    sandbox.disconnect.assert_awaited_once()
    provider.aclose.assert_awaited_once()


async def test_close_does_not_wait_forever_for_a_harness_that_ignores_cancellation(tmp_path, monkeypatch, caplog):
    """A harness that swallows its cancellation gets the shutdown budget, then close disconnects anyway so the
    environment server's own cleanup is not held up; the late worker still answers its turn."""
    agent, _, sandbox, provider, _ = make_agent(tmp_path, monkeypatch)
    agent.config.shutdown_timeout_sec = 0.05
    started, events = asyncio.Event(), []

    class StubbornHarness(FakeHarness):
        async def execute(self, budget):
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                await asyncio.sleep(0.5)
                events.append("harness finished")
                return response_stub(), HarnessOutcome(reason="cancelled"), {}
            raise AssertionError("the turn was not cancelled")

    async def disconnect():
        events.append("disconnect")

    monkeypatch.setattr(module, "MiniSWEHarness", StubbornHarness)
    sandbox.disconnect = AsyncMock(side_effect=disconnect)
    transport = ASGITransport(app=agent.setup_webserver())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.post("/v1/agent_sessions", json=seed_body())).status_code == 200
        turn = asyncio.create_task(client.post(capture_path(), json={"input": "Do it"}))
        await asyncio.wait_for(started.wait(), 5)

        with caplog.at_level("WARNING", logger=module.LOGGER.name):
            close = await client.post("/v1/agent_sessions/close", json=CLOSE_BODY)

        assert close.status_code == 200, close.text
        assert events == ["disconnect"]
        assert any("did not stop within 0.05s" in record.message for record in caplog.records)
        response = await turn
        assert response.status_code == 503 and "cancelled" in response.json()["detail"]
        assert events == ["disconnect", "harness finished"]
    sandbox.disconnect.assert_awaited_once()
    provider.aclose.assert_awaited_once()


async def test_close_waits_for_a_first_turn_still_binding_and_the_turn_never_starts(tmp_path, monkeypatch):
    """A close that arrives while the first turn reads its task context lets that read finish, then stops the
    turn before the harness touches the sandbox, and only then disconnects."""
    agent, _, sandbox, _, _ = make_agent(tmp_path, monkeypatch)
    reading, gate, disconnects_seen = asyncio.Event(), asyncio.Event(), []

    async def blocked_read(sandbox):
        reading.set()
        await gate.wait()
        return TASK_CONTEXT

    class BlockedHarness(FakeHarness):
        async def execute(self, budget):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                disconnects_seen.append(sandbox.disconnect.await_count)
                return response_stub(), HarnessOutcome(reason="cancelled"), {}
            raise AssertionError("the turn was not cancelled")

    monkeypatch.setattr(module, "_read_task_context", blocked_read)
    monkeypatch.setattr(module, "MiniSWEHarness", BlockedHarness)
    transport = ASGITransport(app=agent.setup_webserver())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.post("/v1/agent_sessions", json=seed_body())).status_code == 200
        turn = asyncio.create_task(client.post(capture_path(), json={"input": "Do it"}))
        await asyncio.wait_for(reading.wait(), 5)
        close = asyncio.create_task(client.post("/v1/agent_sessions/close", json=CLOSE_BODY))
        await asyncio.sleep(0.05)
        # The close is waiting for the binding read, with the sandbox still connected.
        assert not close.done() and sandbox.disconnect.await_count == 0

        gate.set()

        assert (await close).status_code == 200
        response = await turn
        assert response.status_code == 503 and "cancelled" in response.json()["detail"]
        # Whether the worker was stopped before its first step or inside the harness, the sandbox was still
        # connected when the harness was cancelled, and no harness ran to completion.
        assert len(FakeHarness.instances) <= 1 and all(count == 0 for count in disconnects_seen)
        sandbox.disconnect.assert_awaited_once()
        # A turn arriving after the close is refused at the door, not bound.
        late = await client.post(capture_path(), json={"input": "Do it"})
        assert late.status_code == 409


async def test_a_closed_session_refuses_to_bind_a_first_turn(tmp_path, monkeypatch):
    agent, _, sandbox, _, _ = make_agent(tmp_path, monkeypatch)
    state = module.MiniSWESession(
        sandbox=sandbox,
        seed=module.SeedSessionResponse(session_id="ag-1", task_id="t", agent_timeout_sec=120),
        original_params=None,
        rollout_id="r1",
        capture_model_calls=True,
        closed=True,
    )
    with pytest.raises(module.HTTPException) as refused:
        await agent._bind_episode_turn(state, module.NeMoGymResponseCreateParamsNonStreaming(input="Do it"), "rk")
    assert refused.value.status_code == 409
    assert state.original_params is None and sandbox.exec.await_count == 0


async def test_a_turn_cancelled_before_its_first_step_is_a_masked_cancellation(tmp_path, monkeypatch):
    """Close can cancel the worker before it ever ran: the waiting turn then answers 503 instead of surfacing a
    cancellation that was never its own."""
    agent, _, sandbox, _, _ = make_agent(tmp_path, monkeypatch)
    params = module.NeMoGymResponseCreateParamsNonStreaming(input="Do it")
    episode = module.AgentSeedSessionRequest.model_validate(seed_body())
    state = module.MiniSWESession(
        sandbox=sandbox,
        seed=module.SeedSessionResponse(session_id="ag-1", task_id="t", agent_timeout_sec=120),
        original_params=params.model_copy(deep=True),
        rollout_id="r1",
        capture_model_calls=False,
        episode=episode,
    )
    agent._sessions[("owner", "r1")] = state

    async def never_ran():
        raise AssertionError("the worker must not run")

    state.worker = asyncio.create_task(never_ran())
    state.worker.cancel()
    request = SimpleNamespace(session={module.SESSION_ID_KEY: "owner"}, cookies={})

    with pytest.raises(module.HTTPException) as refused:
        await agent.responses(request, params)

    assert refused.value.status_code == 503 and "cancelled" in refused.value.detail
    assert state.result is not None and state.result.termination.reason == "cancelled"
    assert state.result.agent_started is False
