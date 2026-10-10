# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import shlex
import shutil
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.agent_utils import process_supervisor
from nemo_gym.agent_utils.sandbox_session import SandboxSession
from nemo_gym.agent_utils.supervisor_client import STOP_REQUEST_FILE, SUPERVISOR_FILE, parse_cleanup_receipt
from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest, _AgentSessionRecord
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_agent import sandbox as hermes_sandbox
from responses_api_agents.hermes_agent.app import (
    HermesAgent,
    HermesAgentConfig,
    HermesAgentRunRequest,
)
from responses_api_agents.hermes_agent.sandbox import HarnessProcessInfo, HermesSandboxSession, parse_runtime_info


@pytest.fixture
def agent(monkeypatch):
    result = HermesAgent(
        config=HermesAgentConfig(
            host="127.0.0.1",
            port=8080,
            name="hermes",
            entrypoint="app.py",
            resources_server={"type": "resources_servers", "name": "resources"},
            model_server={"type": "responses_api_models", "name": "model"},
            enabled_toolsets=["terminal"],
            max_tokens=500,
            temperature=0.7,
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )

    monkeypatch.setattr(
        HermesAgent, "resolve_model_base_url", lambda *args: "http://model:8000/ng-rollout/episode-a1/v1"
    )
    return result


@pytest.fixture
def state():
    sandbox = AsyncMock()
    sandbox.exec.return_value = SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None)
    return HermesSandboxSession(
        request=AgentSeedSessionRequest(
            agent_session_id="session",
            episode_id=EpisodeId(rollout_id="episode", attempt=1),
            task_id=TaskId(taskset="test", task_id="task"),
            sandbox_access={
                "connection": {
                    "kind": "direct",
                    "provider_config_ref": "provider",
                    "descriptor": {"sandbox_id": "task-box"},
                },
                "workdir": "/app",
            },
        ),
        session=SandboxSession(
            sandbox=sandbox, workdir="/app", session_dir="/tmp/nemo-gym-hermes-sessions/session", harness="Hermes"
        ),
    )


def mock_json_download(sandbox, **kwargs):
    """Supply files through the real text utility's sandbox download boundary."""
    download = AsyncMock(**kwargs)

    async def write_download(path, destination):
        Path(destination).write_text(json.dumps(await download(sandbox, path)), encoding="utf-8")

    sandbox.download = AsyncMock(side_effect=write_download)
    return download


def request(state):
    return SimpleNamespace(
        session={"agent_session_id": "session"}, path_params={"rollout_id": state.request.episode_id.capture_key}
    )


def episode(agent):
    body = NeMoGymResponseCreateParamsNonStreaming(input="task")
    return AgentEpisode(
        response=agent._response_from_result(
            body=body,
            result={"completed": True, "messages": [{"role": "assistant", "content": "done"}]},
            model_name="model",
        ),
        observations=AgentObservationBundle(source="hermes"),
    )


async def test_unbound_compatibility_run_fails_before_calling_resources(agent):
    agent.config.resources_server = None
    agent.server_client.post = AsyncMock()
    with pytest.raises(HTTPException, match="/run requires resources_server") as error:
        await agent.run(
            SimpleNamespace(session={}, cookies={}),
            HermesAgentRunRequest(responses_create_params={"input": "task"}),
        )
    assert error.value.status_code == 422
    agent.server_client.post.assert_not_awaited()


async def test_unbound_direct_responses_remains_available(agent):
    agent.config.resources_server = None
    result = episode(agent)
    agent._create_response = AsyncMock(return_value=result.response)
    response = await agent.responses(
        SimpleNamespace(session={}, path_params={}), NeMoGymResponseCreateParamsNonStreaming(input="task")
    )
    assert response == result.response
    agent._create_response.assert_awaited_once()


@pytest.mark.parametrize("bind_resources", [True, False])
def test_http_close_retry_and_stale_activation_never_fall_back(agent, state, bind_resources):
    if not bind_resources:
        agent.config.resources_server = None
    agent._initialize_agent_session_state = AsyncMock(return_value=state)
    agent._run_sandbox_episode = AsyncMock(return_value=episode(agent))
    agent._create_response = AsyncMock(side_effect=AssertionError("host fallback"))
    agent._create_episode = AsyncMock(side_effect=AssertionError("host fallback"))
    with TestClient(agent.setup_webserver()) as client:
        seed = client.post("/v1/agent_sessions", json=state.request.model_dump(mode="json"))
        assert seed.status_code == 200
        assert state.task is None
        assert not state.session.launch_started and state.session.cleanup is None
        assert client.post("/v1/agent_sessions", json=state.request.model_dump(mode="json")).status_code == 200
        path = f"/ng-rollout/{state.request.episode_id.capture_key}/v1/responses"
        activation = client.post(path, json={"input": "task"})
        assert activation.status_code == 200
        assert state.task is not None
        assert state.task.done()
        replay = client.post(path, json={"input": "task"})
        assert replay.status_code == 200
        assert replay.json() == activation.json()
        assert client.post(path, json={"input": "different task"}).status_code == 409
        close = {
            "agent_session_id": seed.json()["agent_session_id"],
            "episode_id": state.request.episode_id.model_dump(),
        }
        first = client.post("/v1/agent_sessions/close", json=close)
        retry = client.post("/v1/agent_sessions/close", json=close)
        assert first.status_code == retry.status_code == 200
        assert agent._session_records["session"].closing
        assert first.json() == retry.json()
        assert client.post(path, json={"input": "task"}).status_code == 409
        assert client.post("/run", json={"responses_create_params": {"input": "task"}}).status_code == 409
        wrong = dict(close, episode_id={"rollout_id": "other"})
        assert client.post("/v1/agent_sessions/close", json=wrong).status_code == 409
    assert agent._run_sandbox_episode.await_count == 1
    state.session.sandbox.disconnect.assert_awaited_once()
    state.session.sandbox.stop.assert_not_awaited()


async def test_invalid_activation_keeps_session_ready(agent: HermesAgent, state: HermesSandboxSession) -> None:
    agent._session_records["session"] = _AgentSessionRecord(state=state, episode_id=state.request.episode_id)
    agent._run_sandbox_episode = AsyncMock(return_value=episode(agent))
    with pytest.raises(HTTPException) as error:
        await agent.responses(request(state), NeMoGymResponseCreateParamsNonStreaming(input="task", top_p=0.9))
    assert error.value.status_code == 422
    assert state.task is None
    agent._run_sandbox_episode.assert_not_awaited()

    await agent.responses(request(state), NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert state.task is not None
    agent._run_sandbox_episode.assert_awaited_once()


async def test_close_cancels_activation_and_its_replay(agent, state):
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def activate(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    agent._session_records["session"] = _AgentSessionRecord(state=state, episode_id=state.request.episode_id)
    agent._run_sandbox_episode = AsyncMock(side_effect=activate)
    body = NeMoGymResponseCreateParamsNonStreaming(input="task")
    running = asyncio.create_task(agent.responses(request(state), body))
    await asyncio.wait_for(started.wait(), 5)
    assert state.task is not None
    with pytest.raises(HTTPException) as error:
        await agent.responses(request(state), body.model_copy(update={"temperature": 0.2}))
    assert error.value.status_code == 409
    replay = asyncio.create_task(agent.responses(request(state), body))
    await asyncio.sleep(0)
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
    first, second = await asyncio.gather(
        agent.close_agent_session(request(state), close), agent.close_agent_session(request(state), close)
    )
    assert first == second
    assert stopped.is_set()
    assert agent._session_records["session"].closing
    with pytest.raises(asyncio.CancelledError):
        await running
    with pytest.raises(asyncio.CancelledError):
        await replay
    agent._run_sandbox_episode.assert_awaited_once()
    state.session.sandbox.disconnect.assert_awaited_once()


async def test_activation_replay_survives_disconnected_waiter(agent, state):
    started = asyncio.Event()
    finish = asyncio.Event()
    expected = episode(agent)

    async def activate(**kwargs):
        started.set()
        await finish.wait()
        return expected

    agent._session_records["session"] = _AgentSessionRecord(state=state, episode_id=state.request.episode_id)
    agent._run_sandbox_episode = AsyncMock(side_effect=activate)
    body = NeMoGymResponseCreateParamsNonStreaming(input="task")
    original = asyncio.create_task(agent.responses(request(state), body))
    await asyncio.wait_for(started.wait(), 5)
    replay = asyncio.create_task(agent.responses(request(state), body.model_copy(deep=True)))
    await asyncio.sleep(0)
    try:
        assert not replay.done()
        original.cancel()
        with pytest.raises(asyncio.CancelledError):
            await original
        assert not state.task.done()
        assert not state.task.cancelling()
        # The request retained for comparison must not alias the caller's mutable model.
        body.input = "changed after dispatch"
        with pytest.raises(HTTPException) as error:
            await agent.responses(request(state), body)
        assert error.value.status_code == 409
        finish.set()
        assert await asyncio.wait_for(replay, 5) == expected.response
        assert await agent.responses(request(state), NeMoGymResponseCreateParamsNonStreaming(input="task")) == (
            expected.response
        )
        agent._run_sandbox_episode.assert_awaited_once()
        assert state.observations == expected.observations
    finally:
        finish.set()
        await asyncio.gather(original, replay, return_exceptions=True)
        await agent.close_agent_session(
            request(state), AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
        )


def test_provider_failure_is_http_500_and_replays_without_rerunning(agent, state):
    agent._initialize_agent_session_state = AsyncMock(return_value=state)
    state.upload_json = AsyncMock()
    mock_json_download(
        state.session.sandbox,
        side_effect=[
            {"cleanup_confirmed": True, "error": None},
            {
                "result": {"failed": True, "error": "HTTP 429 Too Many Requests", "messages": []},
                "runtime": {"hostname": "sandbox", "pid": 123},
            },
        ],
    )
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/v1/agent_sessions", json=state.request.model_dump(mode="json")).status_code == 200
        path = f"/ng-rollout/{state.request.episode_id.capture_key}/v1/responses"
        for _ in range(2):
            response = client.post(path, json={"input": "task"})
            assert response.status_code == 500
        state.upload_json.assert_awaited_once()
        assert state.session.cleanup is not None and state.session.cleanup["cleanup_confirmed"] is True
        close = client.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": "session", "episode_id": state.request.episode_id.model_dump()},
        )
        assert close.status_code == 200
    state.session.sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("runner_started", [False, True], ids=["not-launched", "cleanup-confirmed"])
async def test_close_failure_keeps_session_for_retry(
    agent: HermesAgent, state: HermesSandboxSession, runner_started: bool
) -> None:
    agent._session_records["session"] = _AgentSessionRecord(state=state, episode_id=state.request.episode_id)
    if runner_started:
        state.session.launch_started = True
        download_json = mock_json_download(
            state.session.sandbox, return_value={"cleanup_confirmed": True, "error": None}
        )
    state.session.sandbox.exec.side_effect = [
        SimpleNamespace(return_code=1, error_type=None),
        SimpleNamespace(return_code=0, error_type=None),
    ]
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
    with pytest.raises(RuntimeError, match="session files"):
        await agent.close_agent_session(request(state), close)
    assert agent._session_records["session"].state is state
    assert agent._session_records["session"].closing
    assert (state.session.cleanup is not None) == runner_started
    state.session.sandbox.disconnect.assert_not_awaited()
    with pytest.raises(HTTPException):
        await agent.responses(request(state), NeMoGymResponseCreateParamsNonStreaming(input="task"))
    await agent.close_agent_session(request(state), close)
    state.session.sandbox.disconnect.assert_awaited_once()
    if runner_started:
        download_json.assert_awaited_once()


@pytest.mark.parametrize("receipt", [None, {"cleanup_confirmed": False, "error": "cleanup failed"}])
@pytest.mark.parametrize("launch_started,cleanup", [(False, None), (True, None), (True, {"cleanup_confirmed": True})])
async def test_owned_close_falls_back_to_stop_without_receipt(agent, state, receipt, launch_started, cleanup):
    state.session.owns_sandbox = True
    state.session.launch_started = launch_started
    state.session.cleanup = cleanup
    state.observations = AgentObservationBundle(source="hermes")
    agent._session_records["session"] = _AgentSessionRecord(state=state, episode_id=state.request.episode_id)
    download_json = mock_json_download(state.session.sandbox, return_value=receipt)
    if receipt is None:
        download_json.side_effect = FileNotFoundError("missing cleanup receipt")
    state.session.sandbox.exec.side_effect = AssertionError("Owned close must not depend on sandbox exec")
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)

    response = await agent.close_agent_session(request(state), close)
    assert response.agent_observations == state.observations
    assert state.session.sandbox_stopped
    assert agent._session_records["session"].state is None
    assert await agent.close_agent_session(request(state), close) == response
    state.session.sandbox.stop.assert_awaited_once()
    state.session.sandbox.disconnect.assert_not_awaited()
    if launch_started and cleanup is None:
        state.session.sandbox.exec.assert_awaited_once()
        assert download_json.await_count >= 1
    else:
        state.session.sandbox.exec.assert_not_awaited()
        download_json.assert_not_awaited()


@pytest.mark.parametrize("failure", [RuntimeError, TimeoutError, asyncio.CancelledError])
async def test_owned_stop_failure_keeps_close_retryable(agent, state, failure):
    state.session.owns_sandbox = True
    state.session.launch_started = True
    agent._session_records["session"] = _AgentSessionRecord(state=state, episode_id=state.request.episode_id)
    state.session.sandbox.stop.side_effect = [failure("stop failed"), None]
    download_json = mock_json_download(state.session.sandbox, side_effect=FileNotFoundError("missing cleanup receipt"))
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)

    with pytest.raises(failure):
        await agent.close_agent_session(request(state), close)
    assert agent._session_records["session"].state is state
    assert agent._session_records["session"].closing
    assert state.session.launch_started and state.session.cleanup is None and not state.session.sandbox_stopped
    assert "session" not in agent._closed_session_records
    response = await agent.close_agent_session(request(state), close)
    assert await agent.close_agent_session(request(state), close) == response
    assert state.session.sandbox.stop.await_count == 2
    assert state.session.sandbox_stopped
    state.session.sandbox.disconnect.assert_not_awaited()
    assert state.session.sandbox.exec.await_count == 2
    assert download_json.await_count == 4


async def test_owned_stop_precedes_waiting_for_cancelled_activation(agent, state):
    state.session.owns_sandbox = True
    state.session.launch_started = True
    agent.config.session_close_timeout_seconds = 0.1
    started, stopped = asyncio.Event(), asyncio.Event()

    async def activation():
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            # Model an activation whose cleanup cannot finish until the container stops.
            await stopped.wait()
            raise

    state.session.sandbox.stop.side_effect = stopped.set
    download_json = mock_json_download(state.session.sandbox, side_effect=FileNotFoundError("missing cleanup receipt"))
    state.task = asyncio.create_task(activation())
    await started.wait()
    try:
        await agent._close_agent_session_state(state)
        assert state.task.cancelled()
        assert state.session.sandbox_stopped
        state.session.sandbox.stop.assert_awaited_once()
        state.session.sandbox.exec.assert_awaited_once()
        assert download_json.await_count == 2
    finally:
        stopped.set()
        if not state.task.done():
            state.task.cancel()
        await asyncio.gather(state.task, return_exceptions=True)


@pytest.mark.parametrize("failure", [TimeoutError, asyncio.CancelledError])
async def test_unavailable_remote_fence_still_blocks_close(agent, state, failure):
    # Losing contact with the sandbox is not proof that its pending launch is fenced.
    state.session.sandbox.exec.side_effect = [failure("launch status unavailable"), SimpleNamespace(return_code=0)]
    mock_json_download(state.session.sandbox, side_effect=FileNotFoundError("missing cleanup receipt"))
    state.upload_json = AsyncMock()
    with pytest.raises(failure):
        await agent._run_sandbox_episode(
            body=NeMoGymResponseCreateParamsNonStreaming(input="task"),
            agent_session_id="session",
            state=state,
        )
    assert state.session.launch_started and state.session.cleanup is None and not state.session.sandbox_stopped
    state.session.sandbox.exec.side_effect = None
    with pytest.raises(RuntimeError, match="launch outcome is unknown"):
        await agent._close_agent_session_state(state)
    state.session.sandbox.disconnect.assert_not_awaited()


@pytest.fixture
def local_runner(agent, state, monkeypatch, tmp_path):
    """Execute the actual launch/close shell commands and exchange files, without a remote provider."""
    directory = tmp_path / "session"
    directory.mkdir()
    state.session.session_dir = str(directory)
    state.session.workdir = str(tmp_path)
    agent.config.session_close_timeout_seconds = 2
    monkeypatch.setattr(hermes_sandbox, "_SANDBOX_PYTHON", sys.executable)
    monkeypatch.setattr(
        hermes_sandbox, "_SANDBOX_RUNNER", str(Path(hermes_sandbox.__file__).with_name("sandbox_runner.py"))
    )
    state.session.sandbox.upload.side_effect = shutil.copyfile
    state.session.sandbox.download.side_effect = shutil.copyfile

    async def execute(command, **kwargs):
        process = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            command,
            cwd=kwargs.get("cwd"),
            start_new_session=True,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=8)
            return SimpleNamespace(
                return_code=process.returncode,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                error_type=None,
            )
        finally:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()

    state.session.sandbox.exec.side_effect = execute
    return execute


@pytest.mark.parametrize("failure", ["cancel", "failed-before-spawn"])
async def test_close_fences_a_launch_that_never_reached_the_shell(agent, state, local_runner, failure):
    waiting = asyncio.Event()
    semaphore = asyncio.Semaphore(0)
    commands = []

    async def queued_exec(command, **kwargs):
        if " && exec " in command:
            commands.append(command)
            waiting.set()
            if failure == "cancel":
                await semaphore.acquire()
            raise OSError("provider failed before spawning the shell")
        return await local_runner(command, **kwargs)

    state.session.sandbox.exec.side_effect = queued_exec
    state.task = asyncio.create_task(
        agent._run_sandbox_episode(
            body=NeMoGymResponseCreateParamsNonStreaming(input="task"), agent_session_id="session", state=state
        )
    )
    await asyncio.wait_for(waiting.wait(), timeout=2)
    if failure == "cancel":
        state.task.cancel()
    with pytest.raises(asyncio.CancelledError if failure == "cancel" else OSError):
        await state.task
    assert state.session.cleanup is not None and state.session.cleanup["cleanup_confirmed"] is True
    directory = Path(state.session.session_dir)
    assert (directory / "launch.claim").readlink() == Path("stop")
    assert json.loads((directory / "cleanup.json").read_text())["cleanup_confirmed"] is True

    # A delayed delivery cannot launch, before or after close removes the session directory.
    assert (await local_runner(commands[0])).return_code == 0
    assert not (directory / "supervisor.pid").exists()
    agent._session_records["session"] = _AgentSessionRecord(state=state, episode_id=state.request.episode_id)
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
    first = await agent.close_agent_session(request(state), close)
    assert await agent.close_agent_session(request(state), close) == first
    state.session.sandbox.disconnect.assert_awaited_once()
    assert not directory.exists()
    assert (await local_runner(commands[0])).return_code == 0
    assert not directory.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="The process supervisor requires Linux")
@pytest.mark.parametrize("owned", [False, True])
async def test_close_preserves_interrupted_observations_before_release(
    agent, state, local_runner, monkeypatch, tmp_path, owned
):
    ready = tmp_path / "worker-ready"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json,os,pathlib,signal,sys,time\n"
        "def interrupted(*args):\n"
        "    output = {'result': {'completed': False, 'interrupted': True, 'stop_reason': 'cancelled', "
        "'messages': [{'role': 'assistant', 'content': 'partial answer'}]}, "
        "'observations': {'invocations': [{'invocation_id': 'root', 'status': 'incomplete', "
        "'model_response_ids': ['partial-completion']}]}, "
        "'runtime': {'hostname': 'sandbox', 'pid': os.getpid()}}\n"
        "    pathlib.Path(sys.argv[2]).write_text(json.dumps(output))\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, interrupted)\n"
        f"pathlib.Path({str(ready)!r}).touch()\n"
        "while True: time.sleep(0.01)\n"
    )
    monkeypatch.setattr(hermes_sandbox, "_SANDBOX_RUNNER", str(worker))
    state.session.owns_sandbox = owned
    agent._session_records["session"] = _AgentSessionRecord(state=state, episode_id=state.request.episode_id)
    running = asyncio.create_task(
        agent.responses(request(state), NeMoGymResponseCreateParamsNonStreaming(input="task"))
    )
    try:
        async with asyncio.timeout(5):
            while not ready.exists():
                await asyncio.sleep(0.01)
        close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
        result = await agent.close_agent_session(request(state), close)
        assert result.agent_observations.gaps == []
        invocation = result.agent_observations.records[0]
        assert invocation.status == "incomplete"
        assert invocation.model_calls[0].response_id == "partial-completion"
        assert state.session.artifacts["result"]["messages"][0]["content"] == "partial answer"
        assert state.session.cleanup["cleanup_confirmed"] is True
        assert await agent.close_agent_session(request(state), close) == result
        if owned:
            state.session.sandbox.stop.assert_awaited_once()
        else:
            assert not Path(state.session.session_dir).exists()
            state.session.sandbox.disconnect.assert_awaited_once()
    finally:
        await state.close(2)
        await asyncio.gather(running, return_exceptions=True)


async def test_close_can_recover_a_stop_claim_with_no_receipt(agent, state, local_runner):
    directory = Path(state.session.session_dir)
    # The first close won the claim but was interrupted before publishing its receipt.
    (directory / "launch.claim").symlink_to("stop")
    state.session.launch_started = True
    await state.session.stop_harness(timeout=agent.config.session_close_timeout_seconds)
    assert state.session.cleanup is not None and state.session.cleanup["cleanup_confirmed"] is True
    assert json.loads((directory / "cleanup.json").read_text())["cleanup_confirmed"] is True


@pytest.mark.parametrize("confirmed", [True, False])
async def test_receipt_download_failure_never_signals_a_reused_pid(agent, state, local_runner, tmp_path, confirmed):
    directory = Path(state.session.session_dir)
    signaled = tmp_path / "unrelated-process-signaled"
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import pathlib, signal, sys, time; "
        "signal.signal(signal.SIGTERM, lambda *_: pathlib.Path(sys.argv[1]).touch()); "
        "print('ready', flush=True); time.sleep(30)",
        str(signaled),
        stdout=asyncio.subprocess.PIPE,
    )
    try:
        assert await child.stdout.readline() == b"ready\n"
        (directory / "supervisor.pid").write_text(str(child.pid))
        (directory / "launch.claim").symlink_to("launch")
        (directory / "cleanup.json").write_text(json.dumps({"cleanup_confirmed": confirmed, "error": None}))
        original_download = state.session.sandbox.download
        attempts = 0

        async def transient_download(*args):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("transient download failure")
            return await original_download(*args)

        state.session.sandbox.download = AsyncMock(side_effect=transient_download)
        state.session.launch_started = True
        if confirmed:
            await state.session.stop_harness(timeout=agent.config.session_close_timeout_seconds)
        else:
            with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
                await state.session.stop_harness(timeout=agent.config.session_close_timeout_seconds)
        await asyncio.sleep(0.05)
        assert not signaled.exists()
        assert child.returncode is None
    finally:
        child.kill()
        await child.wait()


@pytest.mark.parametrize(
    "field", ["sandbox_runner_timeout_seconds", "sandbox_install_timeout_seconds", "session_close_timeout_seconds"]
)
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_session_deadlines_are_finite_and_positive(agent, field, value):
    with pytest.raises(ValidationError, match=field):
        HermesAgentConfig.model_validate(agent.config.model_dump() | {field: value})


async def test_borrowed_close_confirms_cleanup_before_cancelling_provider_exec(agent, state):
    entered, confirmed = asyncio.Event(), asyncio.Event()

    async def execute():
        entered.set()
        try:
            await asyncio.Future()
        finally:
            assert confirmed.is_set(), "Provider exec was cancelled before remote cleanup"

    async def terminate(*, timeout):
        assert not state.task.cancelling()
        state.session.cleanup = {"cleanup_confirmed": True, "error": None, "return_code": None, "timed_out": False}
        confirmed.set()

    state.session.stop_harness = AsyncMock(side_effect=terminate)
    state.task = asyncio.create_task(execute())
    await entered.wait()
    try:
        await agent._close_agent_session_state(state)
        assert state.task.cancelled()
        state.session.sandbox.disconnect.assert_awaited_once()
    finally:
        confirmed.set()
        state.task.cancel()
        await asyncio.gather(state.task, return_exceptions=True)


async def test_close_retires_the_launch_path_before_removing_its_fence(
    agent, state, local_runner, monkeypatch, tmp_path
):
    directory = Path(state.session.session_dir)
    retired = Path(f"{state.session.session_dir}.closed")
    state.session.launch_started = True
    commands = tmp_path / "bin"
    commands.mkdir()
    remove = commands / "rm"
    remove.write_text("#!/bin/sh\nexit 1\n")
    remove.chmod(0o755)
    with monkeypatch.context() as patch:
        patch.setenv("PATH", f"{commands}{os.pathsep}{os.environ['PATH']}")
        with pytest.raises(RuntimeError, match="Could not remove Hermes session files"):
            await agent._close_agent_session_state(state)
    assert not directory.exists()
    assert (retired / "launch.claim").readlink() == Path("stop")
    state.session.sandbox.disconnect.assert_not_awaited()
    # Retry completes removal using the stable retired path, without reopening the launch path.
    await agent._close_agent_session_state(state)
    assert not retired.exists()
    state.session.sandbox.disconnect.assert_awaited_once()


async def test_launch_claim_without_pid_is_not_proof_of_cleanup(agent, state, local_runner):
    directory = Path(state.session.session_dir)
    (directory / "launch.claim").symlink_to("launch")
    state.session.launch_started = True
    with pytest.raises(RuntimeError, match="launch outcome is unknown"):
        await agent._close_agent_session_state(state)
    assert state.session.launch_started and state.session.cleanup is None and not state.session.sandbox_stopped
    assert not (directory / "cleanup.json").exists()
    state.session.sandbox.disconnect.assert_not_awaited()


@pytest.mark.parametrize("stop_timing", ["before-shell", "before-handler"])
async def test_stop_during_interpreter_startup_closes_without_starting_a_worker(
    agent, state, local_runner, monkeypatch, tmp_path, stop_timing
):
    directory = Path(state.session.session_dir)
    ready = tmp_path / "before-handler"
    worker_started = tmp_path / "worker-started"
    wrapper = tmp_path / "delayed_supervisor.py"
    wrapper.write_text(
        "import pathlib,runpy,sys,time\n"
        f"if '--help' in sys.argv: runpy.run_path({process_supervisor.__file__!r}, run_name='__main__')\n"
        f"pathlib.Path({str(ready)!r}).touch()\n"
        f"while not pathlib.Path({str(directory / STOP_REQUEST_FILE)!r}).exists(): time.sleep(0.01)\n"
        # Keep the interpreter in the pre-handler window while close sends TERM.
        "time.sleep(0.15)\n"
        f"runner=runpy.run_path({process_supervisor.__file__!r})\n"
        "def unexpected_worker(*args, **kwargs):\n"
        f"    pathlib.Path({str(worker_started)!r}).touch()\n"
        "    raise AssertionError('Worker must not start after the stop marker')\n"
        "runner['subprocess'].Popen=unexpected_worker\n"
        "raise SystemExit(runner['main']())\n"
    )
    state.session.sandbox.upload.side_effect = lambda source, destination: shutil.copyfile(
        wrapper if Path(source).name == SUPERVISOR_FILE else source, destination
    )
    if stop_timing == "before-shell":
        (directory / STOP_REQUEST_FILE).touch()
    state.task = asyncio.create_task(
        agent._run_sandbox_episode(
            body=NeMoGymResponseCreateParamsNonStreaming(input="task"), agent_session_id="session", state=state
        )
    )
    try:
        if stop_timing == "before-handler":
            async with asyncio.timeout(3):
                while not ready.exists():
                    await asyncio.sleep(0.01)
            await state.session.stop_harness(timeout=agent.config.session_close_timeout_seconds)
        with pytest.raises(RuntimeError, match="exited without output"):
            await state.task
        assert state.session.cleanup is not None and state.session.cleanup["cleanup_confirmed"] is True
        assert not worker_started.exists()
        assert json.loads((directory / "cleanup.json").read_text()) == {
            "cleanup_confirmed": True,
            "error": None,
            "return_code": None,
            "timed_out": False,
        }
        await agent._close_agent_session_state(state)
        state.session.sandbox.disconnect.assert_awaited_once()
    finally:
        if not state.task.done():
            state.task.cancel()
        await asyncio.gather(state.task, return_exceptions=True)


def test_cleanup_and_runtime_are_independent():
    cleanup = {"return_code": 0, "timed_out": False, "cleanup_confirmed": True, "error": None}
    assert parse_cleanup_receipt(cleanup) == cleanup
    runtime = HarnessProcessInfo.model_validate({"hostname": "sandbox", "pid": 123})
    assert runtime.hostname == "sandbox" and runtime.pid == 123
    assert runtime.python is None
    with pytest.raises(ValueError, match="cleanup was not confirmed"):
        parse_cleanup_receipt(runtime.model_dump())
    with pytest.raises(ValidationError):
        HarnessProcessInfo.model_validate(cleanup)


@pytest.mark.parametrize("invalid", [{"pid": "123"}, {"hostname": 123}, {"python": 123}, {"return_code": 0}])
def test_runtime_info_keeps_strict_validation(invalid):
    with pytest.raises(ValidationError):
        HarnessProcessInfo.model_validate({"hostname": "sandbox", "pid": 123, **invalid})


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"hostname": "sandbox", "pid": "123"},
        {"hostname": "sandbox", "pid": 123, "python": {"secret": "do not log"}},
    ],
)
def test_malformed_runtime_info_is_optional_and_logs_without_payload(payload, caplog):
    assert parse_runtime_info(payload) is None
    assert "runtime metadata" in caplog.text
    assert "do not log" not in caplog.text


def test_hermes_runtime_uses_the_same_schema():
    payload = {"hostname": "sandbox", "pid": 123, "python": "/opt/hermes/bin/python"}
    assert HarnessProcessInfo.model_validate(payload).model_dump() == payload
    assert parse_runtime_info(payload).model_dump() == payload
    for field in ("hostname", "pid"):
        with pytest.raises(ValidationError):
            HarnessProcessInfo.model_validate({key: value for key, value in payload.items() if key != field})


@pytest.mark.parametrize("runtime", [None, {}, {"hostname": "sandbox", "pid": "123"}])
async def test_invalid_runtime_preserves_response_and_cleanup(agent, state, runtime):
    state.upload_json = AsyncMock()
    mock_json_download(
        state.session.sandbox,
        side_effect=[
            {"cleanup_confirmed": True, "error": None},
            {
                "result": {
                    "completed": True,
                    "messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": "done"}],
                },
                "observations": {"invocations": [{"invocation_id": "root", "model_response_ids": ["completion"]}]},
                "runtime": runtime,
            },
        ],
    )
    result = await agent._run_sandbox_episode(
        body=NeMoGymResponseCreateParamsNonStreaming(input="task"), agent_session_id="session", state=state
    )
    assert result.response.status == "completed"
    assert result.response.output[-1].content[0].text == "done"
    assert result.response.metadata["harness_execution"] == "sandbox"
    assert not {"harness_hostname", "harness_pid", "harness_python"}.intersection(result.response.metadata)
    assert result.observations.records[0].model_calls[0].response_id == "completion"
    assert [gap.code for gap in result.observations.gaps] == ["runtime_info_unavailable"]
    assert state.session.cleanup is not None and state.session.cleanup["cleanup_confirmed"] is True
    close = await agent._close_agent_session_state(state)
    assert close.agent_observations == result.observations
    state.session.sandbox.disconnect.assert_awaited_once()
    state.session.sandbox.stop.assert_not_awaited()


async def test_close_preserves_runtime_gap_after_runner_error(agent, state):
    state.upload_json = AsyncMock()
    mock_json_download(
        state.session.sandbox,
        side_effect=[
            {"cleanup_confirmed": True, "error": None},
            {
                "error": "runner failed",
                "observations": {"invocations": [{"invocation_id": "root", "model_response_ids": ["completion"]}]},
                "runtime": {"hostname": "sandbox", "pid": "invalid"},
            },
        ],
    )
    with pytest.raises(RuntimeError, match="Hermes sandbox runner failed: runner failed"):
        await agent._run_sandbox_episode(
            body=NeMoGymResponseCreateParamsNonStreaming(input="task"), agent_session_id="session", state=state
        )
    close = await agent._close_agent_session_state(state)
    assert close.agent_observations.records[0].model_calls[0].response_id == "completion"
    assert [gap.code for gap in close.agent_observations.gaps] == ["runtime_info_unavailable"]
    state.session.sandbox.disconnect.assert_awaited_once()


async def test_invalid_cleanup_diagnostics_do_not_block_close(agent, state):
    state.session.launch_started = True
    mock_json_download(
        state.session.sandbox,
        return_value={"cleanup_confirmed": True, "error": None, "timed_out": False, "return_code": "0"},
    )
    await agent._close_agent_session_state(state)
    assert state.session.cleanup == {
        "cleanup_confirmed": True,
        "error": None,
        "timed_out": False,
        "return_code": None,
    }
    state.session.sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("receipt", [{}, {"cleanup_confirmed": False}, {"cleanup_confirmed": "true"}])
async def test_runner_exit_without_cleanup_receipt_blocks_close(agent, state, receipt):
    state.session.launch_started = True
    download_json = mock_json_download(state.session.sandbox, return_value=receipt)
    with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
        await agent._close_agent_session_state(state)
    assert state.session.launch_started and state.session.cleanup is None and not state.session.sandbox_stopped
    state.session.sandbox.disconnect.assert_not_awaited()
    download_json.return_value = {"cleanup_confirmed": True, "error": None}
    await agent._close_agent_session_state(state)
    assert state.session.cleanup is not None and state.session.cleanup["cleanup_confirmed"] is True
    state.session.sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("overrides", [{}, {"temperature": 0.0}])
async def test_session_prompt_and_limits_reach_runner(agent, state, overrides, tmp_path, monkeypatch):
    state.session.session_dir = str(tmp_path)
    monkeypatch.setattr(hermes_sandbox, "_SANDBOX_PYTHON", sys.executable)
    state.session.sandbox.upload.side_effect = shutil.copyfile
    agent.server_client.global_config_dict = {
        "model": {"responses_api_models": {"vllm_model": {"chat_template_kwargs": {"enable_thinking": False}}}}
    }
    agent.config.system_prompt = "Configured instruction"
    state.upload_json = AsyncMock()
    mock_json_download(
        state.session.sandbox,
        side_effect=[
            {"cleanup_confirmed": True, "error": None},
            {
                "result": {"completed": True, "messages": [{"role": "assistant", "content": "done"}]},
                "runtime": {"hostname": "sandbox", "pid": 123},
            },
        ],
    )
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="Fix the bug", instructions="Request instruction", **overrides
    )
    await agent._run_sandbox_episode(body=body, agent_session_id="session", state=state)
    launch_command = state.session.sandbox.exec.await_args_list[0].args[0]
    launch_args = shlex.split(launch_command)
    assert any(arg.endswith("/process_supervisor.py") for arg in launch_args)
    assert float(launch_args[launch_args.index("--timeout") + 1]) == agent.config.sandbox_runner_timeout_seconds
    cleanup_timeout = float(launch_args[launch_args.index("--cleanup-timeout") + 1])
    assert cleanup_timeout == agent.config.session_close_timeout_seconds / 3
    assert state.session.sandbox.exec.await_args_list[0].kwargs["timeout_s"] > (
        agent.config.sandbox_runner_timeout_seconds + 3 * cleanup_timeout
    )
    # Execute the real launch prefix: cleanup must receive the shell's PID, not a literal "$".
    launch_prefix, separator, _ = launch_command.partition(" && exec ")
    assert separator
    process = await asyncio.create_subprocess_exec("sh", "-c", launch_prefix)
    assert await process.wait() == 0
    assert int((tmp_path / "supervisor.pid").read_text()) == process.pid
    payload = state.upload_json.await_args.args[1]
    assert payload["stop_request_path"] == state.session.stop_request_path
    assert payload["user_message"] == "Fix the bug"
    assert payload["history"] == []
    assert payload["system_message"] == "Configured instruction\n\nRequest instruction"
    assert payload["max_tokens"] == 500
    assert payload["temperature"] == overrides.get("temperature", 0.7)
    assert body.input == "Fix the bug"  # Do not mutate the caller's request.


@pytest.mark.parametrize("output_available", [True, False])
@pytest.mark.parametrize("hermes_error", [None, "Model generated invalid tool call"])
async def test_exec_reads_final_output_after_confirmed_cleanup(agent, state, output_available, hermes_error):
    state.upload_json = AsyncMock()
    state.session.read_output_log = AsyncMock(return_value="runner stderr")
    events = []

    async def execute(command, **kwargs):
        events.append("exec")
        return SimpleNamespace(stdout="runner stderr", stderr="", return_code=0, error_type=None)

    state.session.sandbox.exec.side_effect = execute

    async def download(sandbox, path):
        if path.endswith("/cleanup.json"):
            events.append("cleanup")
            return {"cleanup_confirmed": True, "error": None}
        assert path.endswith("/output.json")
        events.append("output")
        if not output_available:
            raise FileNotFoundError(path)
        return {
            "result": {
                "completed": hermes_error is None,
                "error": hermes_error,
                "messages": [
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "Patch done"},
                ],
            },
            "observations": {
                "invocations": [
                    {
                        "invocation_id": "root",
                        "status": "failed" if hermes_error else "completed",
                        "model_response_ids": ["completion"],
                    }
                ]
            },
            "runtime": {"hostname": "sandbox", "pid": 123},
        }

    mock_json_download(state.session.sandbox, side_effect=download)
    activation = agent._run_sandbox_episode(
        body=NeMoGymResponseCreateParamsNonStreaming(input="task"), agent_session_id="session", state=state
    )
    if output_available:
        result = await activation
        assert result.response.status == ("failed" if hermes_error else "completed")
        assert result.response.output[-1].content[0].text == "Patch done"
        assert result.observations.records[0].model_calls[0].response_id == "completion"
        assert result.observations.records[0].model_calls[0].model_ref == agent.config.model_server
    else:
        with pytest.raises(RuntimeError, match="runner exited without output: runner stderr"):
            await activation
    assert events[:3] == ["exec", "cleanup", "output"]
    assert state.session.cleanup is not None and state.session.cleanup["cleanup_confirmed"] is True
    agent.server_client.post.assert_not_called()
    payload = state.upload_json.await_args.args[1]
    assert payload["model_base_url"] == "http://model:8000/ng-rollout/episode-a1/v1"


@pytest.mark.parametrize(
    "override",
    [
        {"top_p": 0.8},
        {"top_p": 1.0},
        {"max_output_tokens": 32},
        {"model": "different-model"},
        {"store": True},
        {"store": False},
        {"service_tier": "priority"},
        {"include": ["reasoning.encrypted_content"]},
        {"user": "user-id"},
        {"metadata": {"extra_body": '{"seed": 1}'}},
        {"metadata": {"chat_template_kwargs": '{"enable_thinking": true}'}},
        {"prompt_cache_key": "cache"},
        {"reasoning": {"effort": "low"}},
        {"previous_response_id": "previous"},
        {"tool_choice": "none"},
        {"parallel_tool_calls": False},
        {"background": True},
        {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "https://example.com/x.png", "detail": "auto"}],
                }
            ]
        },
        {"input": [{"type": "function_call_output", "call_id": "call", "output": "result"}]},
    ],
)
@pytest.mark.parametrize("sandbox", [False, True])
async def test_unsupported_requests_are_rejected(agent, state, override, sandbox):
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": "task"} | override)
    agent._session_records["session"] = _AgentSessionRecord(state=state, episode_id=state.request.episode_id)
    agent._run_sandbox_episode = AsyncMock(side_effect=AssertionError("Must validate before execution"))
    agent._create_response = AsyncMock(side_effect=AssertionError("Must validate before execution"))
    with pytest.raises(HTTPException) as error:
        await agent.responses(request(state) if sandbox else SimpleNamespace(session={}), body)
    assert error.value.status_code == 422
    assert state.task is None
    agent._run_sandbox_episode.assert_not_awaited()
    agent._create_response.assert_not_awaited()


@pytest.mark.parametrize("temperature", [None, 0.0, 0.2])
async def test_host_and_sandbox_prepare_the_same_request(agent, state, monkeypatch, temperature):
    validations = []
    validate = HermesAgent._validate_request

    def count_validation(self, body):
        validations.append(body)
        return validate(self, body)

    monkeypatch.setattr(HermesAgent, "_validate_request", count_validation)
    agent.config.system_prompt = "Configured instruction"
    body = NeMoGymResponseCreateParamsNonStreaming(
        model="model",
        instructions="Request instruction",
        temperature=temperature,
        input=[
            {"role": "system", "content": "Input system instruction"},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up"},
        ],
    )
    original = body.model_dump()
    result = {"completed": True, "messages": [{"role": "assistant", "content": "done"}]}
    runner = MagicMock()
    runner.run_conversation.return_value = result
    constructor = MagicMock(return_value=runner)
    monkeypatch.setattr("run_agent.AIAgent", constructor)
    monkeypatch.setattr(HermesAgent, "_ensure_sigterm_handler", lambda *_: None)
    await agent.responses(SimpleNamespace(session={}), body)
    user_message, system_message, history = runner.run_conversation.call_args.args

    agent._session_records["session"] = _AgentSessionRecord(state=state, episode_id=state.request.episode_id)
    state.upload_json = AsyncMock()
    mock_json_download(
        state.session.sandbox,
        side_effect=[
            {"cleanup_confirmed": True, "error": None},
            {"result": result, "runtime": {"hostname": "sandbox", "pid": 123}},
        ],
    )
    await agent.responses(request(state), body)
    payload = state.upload_json.await_args.args[1]
    assert payload["user_message"] == user_message == "Follow-up"
    assert (
        payload["history"]
        == history
        == [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
        ]
    )
    assert (
        payload["system_message"]
        == system_message
        == ("Configured instruction\n\nRequest instruction\n\nInput system instruction")
    )
    assert (
        payload["temperature"]
        == constructor.call_args.kwargs["temperature"]
        == (temperature if temperature is not None else 0.7)
    )
    assert payload["max_tokens"] == constructor.call_args.kwargs["max_tokens"] == 500
    assert body.model_dump() == original
    assert len(validations) == 2  # Exactly once per incoming request, not again in the runner.


def test_future_schema_controls_are_rejected_unless_left_at_default(agent):
    class ExtendedRequest(NeMoGymResponseCreateParamsNonStreaming):
        future_control: str | None = None

    body = ExtendedRequest(input="task", stream=False, background=False)
    assert agent._validate_request(body).input[0].content == "task"
    with pytest.raises(HTTPException, match="future_control"):
        agent._validate_request(body.model_copy(update={"future_control": "enabled"}))


def test_text_history_and_system_message_are_preserved(agent):
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": [{"type": "input_text", "text": "First question"}]},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up"},
        ]
    )
    assert agent._validate_request(body).input == body.input


async def test_required_resources_tools_are_rejected_before_connect(agent, state):
    from nemo_gym.tool_access import DirectHTTPToolAccess

    state.request.tool_accesses = [DirectHTTPToolAccess(name="required", required=True, base_url="http://tools")]
    with pytest.raises(HTTPException, match="only MCP tool grants"):
        await agent.seed_agent_session(SimpleNamespace(session={}), state.request)


async def test_seed_binds_full_payload_and_is_serialized(agent, state):
    agent._initialize_agent_session_state = AsyncMock(return_value=state)
    requests = [SimpleNamespace(session={}), SimpleNamespace(session={})]
    result = await asyncio.gather(*(agent.seed_agent_session(req, state.request) for req in requests))
    assert [item.agent_session_id for item in result] == [state.request.agent_session_id] * 2
    agent._initialize_agent_session_state.assert_awaited_once()
    changed = state.request.model_copy(deep=True)
    changed.sandbox_access.workdir = "/other"
    with pytest.raises(HTTPException, match="another seed"):
        await agent.seed_agent_session(requests[1], changed)
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
    result = await agent.close_agent_session(SimpleNamespace(session={}), close)
    assert await agent.close_agent_session(SimpleNamespace(session={}), close) == result
    assert all(record.state is None for record in agent._session_records.values())
    state.session.sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("marker", [None, "", 0, [], {}])
async def test_malformed_cookie_cannot_fall_back_or_seed(agent, state, marker):
    malformed = SimpleNamespace(session={"agent_session_id": marker})
    agent._create_response = AsyncMock(side_effect=AssertionError("host fallback"))
    with pytest.raises(HTTPException, match="Invalid agent"):
        await agent.responses(malformed, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    with pytest.raises(HTTPException, match="Invalid agent"):
        await agent.seed_agent_session(malformed, state.request)
    with pytest.raises(HTTPException, match="Invalid agent"):
        await agent.close_agent_session(
            malformed, AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
        )
    with pytest.raises(HTTPException, match="Invalid agent"):
        await agent.run(malformed, HermesAgentRunRequest(responses_create_params={"input": "task"}))


@pytest.mark.parametrize("owns_sandbox", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_failed_setup_retains_handle_until_cleanup_confirmed(
    agent, state, monkeypatch, owns_sandbox, cleanup_fails
):
    import responses_api_agents.hermes_agent.app as module

    body = state.request.model_copy(deep=True)
    if owns_sandbox:
        body.sandbox_access = None
        agent.config.sandbox_provider = "runtime"
        agent.config.sandbox_config = {"workdir": "/fallback"}
    sandbox = state.session.sandbox
    factory = MagicMock(return_value=sandbox)
    factory.connect = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(module, "AsyncSandbox", factory)
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(module, "resolve_provider_config", lambda *args: {})
    monkeypatch.setattr(module, "create_provider", lambda config: AsyncMock())
    monkeypatch.setattr(module.shutil, "which", lambda name: "/test/uv")
    ok = SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None)
    failed = SimpleNamespace(return_code=1, stdout="", stderr="installer failed")
    # Prepare paths, detect a missing runtime, probe the architecture for uv, fail installation, then
    # remove session files.
    sandbox.exec.side_effect = [ok, failed, ok, failed, ok]
    cleanup = sandbox.stop if owns_sandbox else sandbox.disconnect
    if cleanup_fails:
        cleanup.side_effect = RuntimeError("cleanup unavailable")
    with pytest.raises(RuntimeError, match="installer failed"):
        await agent.seed_agent_session(SimpleNamespace(session={}), body)
    if cleanup_fails:
        assert agent._session_records["session"].closing
        with pytest.raises(HTTPException, match="closing"):
            await agent.seed_agent_session(SimpleNamespace(session={}), body)
    else:
        assert all(record.state is None for record in agent._session_records.values())
    sandbox.exec.side_effect = None
    cleanup.side_effect = None
    receipt = await agent.close_agent_session(
        SimpleNamespace(session={}),
        AgentCloseSessionRequest(agent_session_id="session", episode_id=body.episode_id),
    )
    assert receipt.agent_session_id == "session"
    assert all(record.state is None for record in agent._session_records.values())
    if owns_sandbox:
        sandbox.disconnect.assert_not_awaited()
    else:
        sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize("url", ["http://127.0.0.1:8000/ng-rollout/native-a1/v1", "http://[::1]:8000/v1"])
async def test_seed_refuses_loopback_model_url_on_remote_provider(agent, state, monkeypatch, url):
    import responses_api_agents.hermes_agent.app as module

    monkeypatch.setattr(HermesAgent, "resolve_model_base_url", lambda *args: url)
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(module, "resolve_provider_config", lambda *args: {"opensandbox": {}})
    provider = AsyncMock()
    create_provider = MagicMock(return_value=provider)
    monkeypatch.setattr(module, "create_provider", create_provider)
    factory = MagicMock(return_value=state.session.sandbox)
    factory.connect = AsyncMock(return_value=state.session.sandbox)
    monkeypatch.setattr(module, "AsyncSandbox", factory)

    with pytest.raises(ValueError, match=r"\+\+use_absolute_ip=true"):
        await agent.seed_agent_session(SimpleNamespace(session={}), state.request)

    # Refused before the sandbox was connected or anything uploaded, so there is nothing to clean up.
    create_provider.assert_not_called()
    factory.connect.assert_not_awaited()
    state.session.sandbox.upload.assert_not_awaited()
    state.session.sandbox.exec.assert_not_awaited()
    assert all(record.state is None for record in agent._session_records.values())


async def test_seed_allows_loopback_model_url_on_local_docker(agent, state, monkeypatch, caplog):
    import responses_api_agents.hermes_agent.app as module

    monkeypatch.setattr(HermesAgent, "resolve_model_base_url", lambda *args: "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(module, "resolve_provider_config", lambda *args: {"docker": {}})
    monkeypatch.setattr(module, "create_provider", lambda config: AsyncMock())
    factory = MagicMock(return_value=state.session.sandbox)
    factory.connect = AsyncMock(return_value=state.session.sandbox)
    monkeypatch.setattr(module, "AsyncSandbox", factory)

    with caplog.at_level("WARNING", logger="nemo_gym.base_responses_api_agent"):
        result = await agent.seed_agent_session(SimpleNamespace(session={}), state.request)

    assert result.agent_session_id == state.request.agent_session_id
    assert sum("use_absolute_ip" in record.getMessage() for record in caplog.records) == 1
    factory.connect.assert_awaited_once()
    assert state.session.sandbox.upload.await_count >= 3
