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

from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest, _AgentSessionRecord
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_agent import app as hermes_app
from responses_api_agents.hermes_agent.app import (
    HermesAgent,
    HermesAgentConfig,
    HermesAgentRunRequest,
    HermesAgentSessionState,
    RunnerCleanup,
)


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
        HermesAgent, "resolve_model_base_url", lambda *args: "http://model:8000/ng-rollout/native-a1/v1"
    )
    return result


@pytest.fixture
def state():
    sandbox = AsyncMock()
    sandbox.exec.return_value = SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None)
    return HermesAgentSessionState(
        request=AgentSeedSessionRequest(
            agent_session_id="session",
            episode_id=EpisodeId(rollout_id="native", attempt=1),
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
        sandbox=sandbox,
        workdir="/app",
        session_dir="/tmp/nemo-gym-hermes-sessions/session",
    )


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


def test_http_close_retry_and_stale_activation_never_fall_back(agent, state):
    agent._initialize_agent_session_state = AsyncMock(return_value=state)
    agent._run_sandbox_episode = AsyncMock(return_value=episode(agent))
    agent._create_response = AsyncMock(side_effect=AssertionError("host fallback"))
    agent._create_episode = AsyncMock(side_effect=AssertionError("host fallback"))
    with TestClient(agent.setup_webserver()) as client:
        seed = client.post("/v1/agent_sessions", json=state.request.model_dump(mode="json"))
        assert seed.status_code == 200
        assert state.task is None
        assert state.runner_cleanup is RunnerCleanup.IDLE
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
    state.sandbox.disconnect.assert_awaited_once()
    state.sandbox.stop.assert_not_awaited()


async def test_invalid_activation_keeps_session_ready(agent: HermesAgent, state: HermesAgentSessionState) -> None:
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
    state.sandbox.disconnect.assert_awaited_once()


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
    agent._upload_json = AsyncMock()
    agent._download_json = AsyncMock(
        side_effect=[
            {"cleanup_confirmed": True},
            {"result": {"failed": True, "error": "HTTP 429 Too Many Requests", "messages": []}, "runtime": {}},
        ]
    )
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/v1/agent_sessions", json=state.request.model_dump(mode="json")).status_code == 200
        path = f"/ng-rollout/{state.request.episode_id.capture_key}/v1/responses"
        for _ in range(2):
            response = client.post(path, json={"input": "task"})
            assert response.status_code == 500
        agent._upload_json.assert_awaited_once()
        assert state.runner_cleanup is RunnerCleanup.CONFIRMED
        close = client.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": "session", "episode_id": state.request.episode_id.model_dump()},
        )
        assert close.status_code == 200
    state.sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("runner_started", [False, True], ids=["not-launched", "cleanup-confirmed"])
async def test_close_failure_keeps_session_for_retry(
    agent: HermesAgent, state: HermesAgentSessionState, runner_started: bool
) -> None:
    agent._session_records["session"] = _AgentSessionRecord(state=state, episode_id=state.request.episode_id)
    if runner_started:
        state.runner_cleanup = RunnerCleanup.UNCONFIRMED
        agent._download_json = AsyncMock(return_value={"cleanup_confirmed": True})
    state.sandbox.exec.side_effect = [SimpleNamespace(return_code=1), SimpleNamespace(return_code=0)]
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
    with pytest.raises(RuntimeError, match="session files"):
        await agent.close_agent_session(request(state), close)
    assert agent._session_records["session"].state is state
    assert agent._session_records["session"].closing
    assert state.runner_cleanup is (RunnerCleanup.CONFIRMED if runner_started else RunnerCleanup.IDLE)
    state.sandbox.disconnect.assert_not_awaited()
    with pytest.raises(HTTPException):
        await agent.responses(request(state), NeMoGymResponseCreateParamsNonStreaming(input="task"))
    await agent.close_agent_session(request(state), close)
    state.sandbox.disconnect.assert_awaited_once()
    if runner_started:
        agent._download_json.assert_awaited_once()


@pytest.mark.parametrize("receipt", [None, {"cleanup_confirmed": False, "error": "cleanup failed"}])
@pytest.mark.parametrize("runner_cleanup", list(RunnerCleanup))
async def test_owned_close_stops_without_receipt_or_filesystem_cleanup(agent, state, receipt, runner_cleanup):
    state.owns_sandbox = True
    state.runner_cleanup = runner_cleanup
    state.observations = AgentObservationBundle(source="hermes")
    agent._session_records["session"] = _AgentSessionRecord(state=state, episode_id=state.request.episode_id)
    agent._download_json = AsyncMock(return_value=receipt)
    if receipt is None:
        agent._download_json.side_effect = FileNotFoundError("missing cleanup receipt")
    state.sandbox.exec.side_effect = AssertionError("Owned close must not depend on sandbox exec")
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)

    response = await agent.close_agent_session(request(state), close)
    assert response.agent_observations == state.observations
    assert state.runner_cleanup is RunnerCleanup.CONFIRMED
    assert agent._session_records["session"].state is None
    assert await agent.close_agent_session(request(state), close) == response
    state.sandbox.stop.assert_awaited_once()
    state.sandbox.disconnect.assert_not_awaited()
    state.sandbox.exec.assert_not_awaited()
    agent._download_json.assert_not_awaited()


@pytest.mark.parametrize("failure", [RuntimeError, TimeoutError, asyncio.CancelledError])
async def test_owned_stop_failure_keeps_close_retryable(agent, state, failure):
    state.owns_sandbox = True
    state.runner_cleanup = RunnerCleanup.UNCONFIRMED
    agent._session_records["session"] = _AgentSessionRecord(state=state, episode_id=state.request.episode_id)
    state.sandbox.stop.side_effect = [failure("stop failed"), None]
    agent._download_json = AsyncMock(side_effect=FileNotFoundError("missing cleanup receipt"))
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)

    with pytest.raises(failure, match="stop failed"):
        await agent.close_agent_session(request(state), close)
    assert agent._session_records["session"].state is state
    assert agent._session_records["session"].closing
    assert state.runner_cleanup is RunnerCleanup.UNCONFIRMED
    assert "session" not in agent._closed_session_records
    response = await agent.close_agent_session(request(state), close)
    assert await agent.close_agent_session(request(state), close) == response
    assert state.sandbox.stop.await_count == 2
    assert state.runner_cleanup is RunnerCleanup.CONFIRMED
    state.sandbox.disconnect.assert_not_awaited()
    state.sandbox.exec.assert_not_awaited()
    agent._download_json.assert_not_awaited()


async def test_owned_stop_precedes_waiting_for_cancelled_activation(agent, state):
    state.owns_sandbox = True
    state.runner_cleanup = RunnerCleanup.UNCONFIRMED
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

    state.sandbox.stop.side_effect = stopped.set
    agent._download_json = AsyncMock(side_effect=FileNotFoundError("missing cleanup receipt"))
    state.task = asyncio.create_task(activation())
    await started.wait()
    try:
        await agent._close_agent_session_state(state)
        assert state.task.cancelled()
        assert state.runner_cleanup is RunnerCleanup.CONFIRMED
        state.sandbox.stop.assert_awaited_once()
        state.sandbox.exec.assert_not_awaited()
        agent._download_json.assert_not_awaited()
    finally:
        stopped.set()
        if not state.task.done():
            state.task.cancel()
        await asyncio.gather(state.task, return_exceptions=True)


@pytest.mark.parametrize("failure", [TimeoutError, asyncio.CancelledError])
async def test_unavailable_remote_fence_still_blocks_close(agent, state, failure):
    # Losing contact with the sandbox is not proof that its pending launch is fenced.
    state.sandbox.exec.side_effect = [failure("launch status unavailable"), SimpleNamespace(return_code=0)]
    agent._download_json = AsyncMock(side_effect=FileNotFoundError("missing cleanup receipt"))
    agent._upload_json = AsyncMock()
    with pytest.raises(failure):
        await agent._run_sandbox_episode(
            body=NeMoGymResponseCreateParamsNonStreaming(input="task"),
            agent_session_id="session",
            state=state,
        )
    assert state.runner_cleanup is RunnerCleanup.UNCONFIRMED
    state.sandbox.exec.side_effect = None
    with pytest.raises(RuntimeError, match="launch outcome is unknown"):
        await agent._close_agent_session_state(state)
    state.sandbox.disconnect.assert_not_awaited()


@pytest.fixture
def local_runner(agent, state, monkeypatch, tmp_path):
    """Execute the actual launch/close shell commands and exchange files, without a remote provider."""
    directory = tmp_path / "session"
    directory.mkdir()
    state.session_dir = str(directory)
    state.workdir = str(tmp_path)
    agent.config.session_close_timeout_seconds = 2
    monkeypatch.setattr(hermes_app, "_SANDBOX_PYTHON", sys.executable)
    monkeypatch.setattr(hermes_app, "_SANDBOX_RUNNER", str(Path(hermes_app.__file__).with_name("sandbox_runner.py")))
    monkeypatch.setattr(hermes_app, "_SANDBOX_SUPERVISOR", hermes_app.process_supervisor.__file__)
    state.sandbox.upload.side_effect = shutil.copyfile
    state.sandbox.download.side_effect = shutil.copyfile

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

    state.sandbox.exec.side_effect = execute
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

    state.sandbox.exec.side_effect = queued_exec
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
    assert state.runner_cleanup is RunnerCleanup.CONFIRMED
    directory = Path(state.session_dir)
    assert (directory / "launch.claim").readlink() == Path("stop")
    assert json.loads((directory / "cleanup.json").read_text())["cleanup_confirmed"] is True

    # A delayed delivery cannot launch, before or after close removes the session directory.
    assert (await local_runner(commands[0])).return_code == 0
    assert not (directory / "runner.pid").exists()
    agent._session_records["session"] = _AgentSessionRecord(state=state, episode_id=state.request.episode_id)
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
    first = await agent.close_agent_session(request(state), close)
    assert await agent.close_agent_session(request(state), close) == first
    state.sandbox.disconnect.assert_awaited_once()
    assert not directory.exists()
    assert (await local_runner(commands[0])).return_code == 0
    assert not directory.exists()


async def test_close_can_recover_a_stop_claim_with_no_receipt(agent, state, local_runner):
    directory = Path(state.session_dir)
    # The first close won the claim but was interrupted before publishing its receipt.
    (directory / "launch.claim").symlink_to("stop")
    state.runner_cleanup = RunnerCleanup.UNCONFIRMED
    await agent._terminate_sandbox_runner(state)
    assert state.runner_cleanup is RunnerCleanup.CONFIRMED
    assert json.loads((directory / "cleanup.json").read_text())["cleanup_confirmed"] is True


@pytest.mark.parametrize("confirmed", [True, False])
async def test_receipt_download_failure_never_signals_a_reused_pid(agent, state, local_runner, tmp_path, confirmed):
    directory = Path(state.session_dir)
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
        (directory / "runner.pid").write_text(str(child.pid))
        (directory / "launch.claim").symlink_to("launch")
        (directory / "cleanup.json").write_text(json.dumps({"cleanup_confirmed": confirmed}))
        download = agent._download_json
        attempts = 0

        async def transient_download(*args):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("transient download failure")
            return await download(*args)

        agent._download_json = AsyncMock(side_effect=transient_download)
        state.runner_cleanup = RunnerCleanup.UNCONFIRMED
        if confirmed:
            await agent._terminate_sandbox_runner(state)
        else:
            with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
                await agent._terminate_sandbox_runner(state)
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

    async def terminate(_state):
        assert not state.task.cancelling()
        state.runner_cleanup = RunnerCleanup.CONFIRMED
        confirmed.set()

    agent._terminate_sandbox_runner = AsyncMock(side_effect=terminate)
    state.task = asyncio.create_task(execute())
    await entered.wait()
    try:
        await agent._close_agent_session_state(state)
        assert state.task.cancelled()
        state.sandbox.disconnect.assert_awaited_once()
    finally:
        confirmed.set()
        state.task.cancel()
        await asyncio.gather(state.task, return_exceptions=True)


async def test_close_retires_the_launch_path_before_removing_its_fence(
    agent, state, local_runner, monkeypatch, tmp_path
):
    directory = Path(state.session_dir)
    retired = Path(f"{state.session_dir}.closed")
    state.runner_cleanup = RunnerCleanup.UNCONFIRMED
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
    state.sandbox.disconnect.assert_not_awaited()
    # Retry completes removal using the stable retired path, without reopening the launch path.
    await agent._close_agent_session_state(state)
    assert not retired.exists()
    state.sandbox.disconnect.assert_awaited_once()


async def test_launch_claim_without_pid_is_not_proof_of_cleanup(agent, state, local_runner):
    directory = Path(state.session_dir)
    (directory / "launch.claim").symlink_to("launch")
    state.runner_cleanup = RunnerCleanup.UNCONFIRMED
    with pytest.raises(RuntimeError, match="launch outcome is unknown"):
        await agent._close_agent_session_state(state)
    assert state.runner_cleanup is RunnerCleanup.UNCONFIRMED
    assert not (directory / "cleanup.json").exists()
    state.sandbox.disconnect.assert_not_awaited()


@pytest.mark.parametrize("stop_timing", ["before-shell", "before-handler"])
async def test_stop_during_interpreter_startup_closes_without_starting_a_worker(
    agent, state, local_runner, monkeypatch, tmp_path, stop_timing
):
    directory = Path(state.session_dir)
    ready = tmp_path / "before-handler"
    worker_started = tmp_path / "worker-started"
    wrapper = tmp_path / "delayed_supervisor.py"
    wrapper.write_text(
        "import pathlib,runpy,sys,time\n"
        f"pathlib.Path({str(ready)!r}).touch()\n"
        f"while not pathlib.Path({str(directory / 'runner.stop')!r}).exists(): time.sleep(0.01)\n"
        # Keep the interpreter in the pre-handler window while close sends TERM.
        "time.sleep(0.15)\n"
        f"runner=runpy.run_path({hermes_app._SANDBOX_SUPERVISOR!r})\n"
        "def unexpected_worker(*args, **kwargs):\n"
        f"    pathlib.Path({str(worker_started)!r}).touch()\n"
        "    raise AssertionError('Worker must not start after the stop marker')\n"
        "runner['subprocess'].Popen=unexpected_worker\n"
        "raise SystemExit(runner['main']())\n"
    )
    monkeypatch.setattr(hermes_app, "_SANDBOX_SUPERVISOR", str(wrapper))
    if stop_timing == "before-shell":
        (directory / "runner.stop").touch()
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
            await agent._terminate_sandbox_runner(state)
        with pytest.raises(RuntimeError, match="exited without output"):
            await state.task
        assert state.runner_cleanup is RunnerCleanup.CONFIRMED
        assert not worker_started.exists()
        assert json.loads((directory / "cleanup.json").read_text()) == {
            "cleanup_confirmed": True,
            "error": None,
            "return_code": None,
            "timed_out": False,
        }
        await agent._close_agent_session_state(state)
        state.sandbox.disconnect.assert_awaited_once()
    finally:
        if not state.task.done():
            state.task.cancel()
        await asyncio.gather(state.task, return_exceptions=True)


@pytest.mark.parametrize("receipt", [{}, {"cleanup_confirmed": False}, {"cleanup_confirmed": "true"}])
async def test_runner_exit_without_cleanup_receipt_blocks_close(agent, state, receipt):
    state.runner_cleanup = RunnerCleanup.UNCONFIRMED
    agent._download_json = AsyncMock(return_value=receipt)
    with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
        await agent._close_agent_session_state(state)
    assert state.runner_cleanup is RunnerCleanup.UNCONFIRMED
    state.sandbox.disconnect.assert_not_awaited()
    agent._download_json.return_value = {"cleanup_confirmed": True}
    await agent._close_agent_session_state(state)
    assert state.runner_cleanup is RunnerCleanup.CONFIRMED
    state.sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("overrides", [{}, {"temperature": 0.0}])
async def test_native_prompt_and_limits_reach_runner(agent, state, overrides, tmp_path):
    state.session_dir = str(tmp_path)
    agent.server_client.global_config_dict = {
        "model": {"responses_api_models": {"vllm_model": {"chat_template_kwargs": {"enable_thinking": False}}}}
    }
    agent.config.system_prompt = "Configured instruction"
    agent._upload_json = AsyncMock()
    agent._download_json = AsyncMock(
        side_effect=[
            {"cleanup_confirmed": True},
            {
                "result": {"completed": True, "messages": [{"role": "assistant", "content": "done"}]},
                "runtime": {"pid": 123},
            },
        ]
    )
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="Fix the bug", instructions="Request instruction", **overrides
    )
    await agent._run_sandbox_episode(body=body, agent_session_id="session", state=state)
    launch_command = state.sandbox.exec.await_args_list[0].args[0]
    launch_args = shlex.split(launch_command)
    assert any(arg.endswith("/process_supervisor.py") for arg in launch_args)
    assert float(launch_args[launch_args.index("--timeout") + 1]) == agent.config.sandbox_runner_timeout_seconds
    cleanup_timeout = float(launch_args[launch_args.index("--cleanup-timeout") + 1])
    assert cleanup_timeout == agent.config.session_close_timeout_seconds / 3
    assert state.sandbox.exec.await_args_list[0].kwargs["timeout_s"] > (
        agent.config.sandbox_runner_timeout_seconds + 3 * cleanup_timeout
    )
    # Execute the real launch prefix: cleanup must receive the shell's PID, not a literal "$".
    launch_prefix, separator, _ = launch_command.partition(" && exec ")
    assert separator
    process = await asyncio.create_subprocess_exec("sh", "-c", launch_prefix)
    assert await process.wait() == 0
    assert int((tmp_path / "runner.pid").read_text()) == process.pid
    payload = agent._upload_json.await_args.args[2]
    assert payload["user_message"] == "Fix the bug"
    assert payload["history"] == []
    assert payload["system_message"] == "Configured instruction\n\nRequest instruction"
    assert payload["max_tokens"] == 500
    assert payload["temperature"] == overrides.get("temperature", 0.7)
    assert body.input == "Fix the bug"  # Do not mutate the caller's request.


@pytest.mark.parametrize("output_available", [True, False])
@pytest.mark.parametrize("hermes_error", [None, "Model generated invalid tool call"])
async def test_exec_reads_final_output_after_confirmed_cleanup(agent, state, output_available, hermes_error):
    agent._upload_json = AsyncMock()
    events = []

    async def execute(command, **kwargs):
        events.append("exec")
        return SimpleNamespace(stdout="runner stderr", stderr="", return_code=0, error_type=None)

    state.sandbox.exec.side_effect = execute

    async def download(sandbox, path):
        if path.endswith("/cleanup.json"):
            events.append("cleanup")
            return {"cleanup_confirmed": True}
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
            "runtime": {"pid": 123},
        }

    agent._download_json = AsyncMock(side_effect=download)
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
    assert state.runner_cleanup is RunnerCleanup.CONFIRMED
    agent.server_client.post.assert_not_called()
    payload = agent._upload_json.await_args.args[2]
    assert payload["model_base_url"] == "http://model:8000/ng-rollout/native-a1/v1"


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
    agent._upload_json = AsyncMock()
    agent._download_json = AsyncMock(side_effect=[{"cleanup_confirmed": True}, {"result": result, "runtime": {}}])
    await agent.responses(request(state), body)
    payload = agent._upload_json.await_args.args[2]
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
    state.sandbox.disconnect.assert_awaited_once()


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
    sandbox = state.sandbox
    factory = MagicMock(return_value=sandbox)
    factory.connect = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(module, "AsyncSandbox", factory)
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(module, "resolve_provider_config", lambda *args: {})
    monkeypatch.setattr(module, "create_provider", lambda config: AsyncMock())
    monkeypatch.setattr(module.shutil, "which", lambda name: "/test/uv")
    ok = SimpleNamespace(return_code=0, stdout="", stderr="")
    failed = SimpleNamespace(return_code=1, stdout="", stderr="installer failed")
    # Prepare paths, detect a missing runtime, fail installation, then remove session files.
    sandbox.exec.side_effect = [ok, failed, failed, ok]
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
