# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real session bindings, activation retries and close receipts."""

import asyncio
import json
import socket
import sys
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from nemo_gym.agents import AGENT_SESSION_COOKIE_KEY, AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.core import EpisodeId, NeMoGymResponseCreateParamsNonStreaming, TaskId
from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess
from responses_api_agents.wildclawbench_agent.app import WildClawAgent
from responses_api_agents.wildclawbench_agent.execution import ExecutionConfig, NativeProcess
from responses_api_agents.wildclawbench_agent.tests.helpers import make_agent


def request(rollout="3-2", session=None):
    return Request(
        dict(
            type="http",
            headers=[],
            path_params={"rollout_id": rollout},
            session=session if session is not None else {},
        )
    )


@pytest.fixture
async def seeded(tmp_path):
    agent = make_agent(tmp_path, sys.executable)
    execution = ExecutionConfig.model_validate(agent.config.model_dump())
    output = tmp_path / "episodes" / ("a" * 64)
    output.mkdir(parents=True)
    seed = AgentSeedSessionRequest(
        agent_session_id="agent-1",
        episode_id=EpisodeId(rollout_id="3-2"),
        task_id=TaskId(taskset="wildclaw_agent", task_id="task"),
        agent_context={"values": {"execution_key": output.name, "owner_host": socket.gethostname()}},
    )
    lease = dict(
        execution=execution.model_dump(mode="json"),
        owner_host=socket.gethostname(),
        metadata=dict(task_id="task", timeout_seconds=300),
        prompt="original prompt",
        episode_id=seed.episode_id.model_dump(mode="json"),
        task_id=seed.task_id.model_dump(mode="json"),
    )
    (output / "lease.json").write_text(json.dumps(lease))
    req = request()
    with patch.object(WildClawAgent, "resolve_model_base_url", return_value="http://policy/rollouts/3-2/v1"):
        await agent.seed_agent_session(req, seed)
    yield agent, seed, req, output
    await agent.close_agent_session(
        req, AgentCloseSessionRequest(agent_session_id=seed.agent_session_id, episode_id=seed.episode_id)
    )


async def test_seed_is_idempotent_and_close_before_activation_keeps_observation_gap(seeded):
    agent, seed, req, output = seeded
    state = agent._require_agent_session(seed.agent_session_id)
    await agent.seed_agent_session(req, seed)
    assert agent._require_agent_session(seed.agent_session_id) is state
    close = AgentCloseSessionRequest(agent_session_id=seed.agent_session_id, episode_id=seed.episode_id)
    first = await agent.close_agent_session(req, close)
    assert first == await agent.close_agent_session(req, close)
    assert first.cleanup_confirmed
    assert first.agent_observations.gaps
    assert (output / "agent_closed.json").is_file()
    assert not (output / "resources_closed.json").exists()
    with pytest.raises(HTTPException) as conflict:
        await agent.responses(
            req, NeMoGymResponseCreateParamsNonStreaming(input=[dict(role="user", content="original prompt")])
        )
    assert conflict.value.status_code == 409


async def test_activation_binds_before_await_shares_retries_and_close_owns_cancellation(seeded, monkeypatch):
    agent, seed, req, output = seeded
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def invoke(self, stage, timeout):
        calls.append(stage)
        entered.set()
        await release.wait()
        return {"events": [], "errors": []}

    monkeypatch.setattr(NativeProcess, "invoke", invoke)
    body = NeMoGymResponseCreateParamsNonStreaming(input=[dict(role="user", content="original prompt")])
    first = asyncio.create_task(agent.responses(req, body))
    await entered.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not agent._require_agent_session(seed.agent_session_id).task.cancelled()
    second = asyncio.create_task(agent.responses(req, body))
    changed = body.model_copy(update={"input": "changed"})
    with pytest.raises(HTTPException) as conflict:
        await agent.responses(req, changed)
    assert conflict.value.status_code == 409
    release.set()
    response = await second
    assert response == await agent.responses(req, body)
    assert calls == ["agent"]
    assert not (output / "resources_closed.json").exists()


@pytest.mark.parametrize(
    "mode", ["missing-cookie", "unknown-cookie", "wrong-route", "changed-prompt", "system", "sampling"]
)
async def test_activation_rejects_unbound_or_modified_inputs(seeded, mode):
    agent, seed, req, _ = seeded
    body = NeMoGymResponseCreateParamsNonStreaming(input=[dict(role="user", content="original prompt")])
    if mode == "missing-cookie":
        req = request()
    elif mode == "unknown-cookie":
        req = request(session={AGENT_SESSION_COOKIE_KEY: "unknown"})
    elif mode == "wrong-route":
        req = request(rollout="4-2", session=req.session)
    elif mode == "changed-prompt":
        body.input[0].content = "modified"
    elif mode == "system":
        body = NeMoGymResponseCreateParamsNonStreaming(input=[dict(role="system", content="original prompt")])
    else:
        body.temperature = 0.7
    with pytest.raises(HTTPException) as refused:
        await agent.responses(req, body)
    assert refused.value.status_code == (409 if mode in {"missing-cookie", "unknown-cookie", "wrong-route"} else 422)
    assert agent._require_agent_session(seed.agent_session_id).task is None


@pytest.mark.parametrize("receipt_name", ["harness_stopped.json", "resources_closed.json"])
async def test_close_cancels_activation_and_retains_state_when_termination_unconfirmed(
    seeded, monkeypatch, receipt_name
):
    agent, seed, req, output = seeded
    entered = asyncio.Event()

    async def invoke(self, stage, timeout):
        (output / "agent_started.json").write_text("{}")
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(NativeProcess, "invoke", invoke)
    activation = asyncio.create_task(
        agent.responses(
            req, NeMoGymResponseCreateParamsNonStreaming(input=[dict(role="user", content="original prompt")])
        )
    )
    await entered.wait()
    close = AgentCloseSessionRequest(agent_session_id=seed.agent_session_id, episode_id=seed.episode_id)
    with pytest.raises(FileNotFoundError):
        await agent.close_agent_session(req, close)
    assert agent._session_records[seed.agent_session_id].state is not None
    with pytest.raises(asyncio.CancelledError):
        await activation
    (output / receipt_name).write_text('{"cleanup_confirmed":true}')
    assert (await agent.close_agent_session(req, close)).cleanup_confirmed


def test_borrowed_sandbox_is_refused_through_real_route_without_image_setup(tmp_path, monkeypatch):
    from nemo_gym.agents import check_agent_seed_rejected

    agent = make_agent(tmp_path, sys.executable)
    seed = AgentSeedSessionRequest(
        agent_session_id="borrowed",
        episode_id=EpisodeId(rollout_id="probe"),
        task_id=TaskId(taskset="probe", task_id="probe"),
        sandbox_access=SandboxAccess(
            connection=DirectSandboxConnection(provider_config_ref="unused", descriptor={}), workdir="/work"
        ),
    )

    def unexpected(*args):
        raise AssertionError("A rejected sandbox must not initialize native assets")

    monkeypatch.setattr("responses_api_agents.wildclawbench_agent.app.image_identity", unexpected)
    assert "does not support sandbox access" in check_agent_seed_rejected(agent.setup_webserver(), seed)


@pytest.mark.parametrize("previous_stage", [False, True])
async def test_cancel_during_spawn_acquires_and_reaps_the_real_child(tmp_path, monkeypatch, previous_stage):
    import contextlib
    import os

    from responses_api_agents.wildclawbench_agent.execution import NativeProcess

    script = tmp_path / "worker-python"
    script.write_text(f"#!{sys.executable}\nimport time;time.sleep(120)\n")
    script.chmod(0o700)
    agent = make_agent(tmp_path, str(script))
    output = tmp_path / "native"
    output.mkdir()
    (output / "worker.json").write_text("{}")
    native = NativeProcess(ExecutionConfig.model_validate(agent.config.model_dump()), output, "sha256:" + "a" * 64)
    if previous_stage:
        native.process = await asyncio.create_subprocess_exec(sys.executable, "-c", "pass")
        await native.process.wait()
    entered, release = asyncio.Event(), asyncio.Event()
    children = []
    original = asyncio.create_subprocess_exec

    async def delayed_handle(*args, **kwargs):
        child = await original(*args, **kwargs)
        children.append(child)
        entered.set()
        await release.wait()
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_handle)
    task = asyncio.create_task(native.invoke("agent", 300))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done(), "Cancellation returned before acquiring the already spawned worker"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
        assert native.process is children[0] and native.process.returncode is not None
        with pytest.raises(ProcessLookupError):
            os.kill(children[0].pid, 0)
    finally:
        release.set()
        for child in children:
            if child.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, 9)
                await child.wait()
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
