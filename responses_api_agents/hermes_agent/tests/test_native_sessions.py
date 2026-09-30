# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import shutil
import signal
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
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
    SessionPhase,
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
            fail_on_error=False,
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
        assert state.phase is SessionPhase.READY
        assert state.runner_cleanup is RunnerCleanup.IDLE
        assert client.post("/v1/agent_sessions", json=state.request.model_dump(mode="json")).status_code == 200
        path = f"/ng-rollout/{state.request.episode_id.capture_key}/v1/responses"
        assert client.post(path, json={"input": "task"}).status_code == 200
        assert state.task is not None
        assert state.task.done()
        assert state.phase is SessionPhase.ACTIVATED
        assert client.post(path, json={"input": "task"}).status_code == 409
        close = {
            "agent_session_id": seed.json()["agent_session_id"],
            "episode_id": state.request.episode_id.model_dump(),
        }
        first = client.post("/v1/agent_sessions/close", json=close)
        retry = client.post("/v1/agent_sessions/close", json=close)
        assert first.status_code == retry.status_code == 200
        assert state.phase is SessionPhase.CLOSING
        assert first.json() == retry.json()
        assert client.post(path, json={"input": "task"}).status_code == 409
        assert client.post("/run", json={"responses_create_params": {"input": "task"}}).status_code == 409
        wrong = dict(close, episode_id={"rollout_id": "other"})
        assert client.post("/v1/agent_sessions/close", json=wrong).status_code == 409
    assert agent._run_sandbox_episode.await_count == 1
    state.sandbox.disconnect.assert_awaited_once()
    state.sandbox.stop.assert_not_awaited()


async def test_invalid_activation_keeps_session_ready(agent: HermesAgent, state: HermesAgentSessionState) -> None:
    agent._agent_sessions["session"] = state
    agent._run_sandbox_episode = AsyncMock(return_value=episode(agent))
    with pytest.raises(HTTPException) as error:
        await agent.responses(request(state), NeMoGymResponseCreateParamsNonStreaming(input="task", top_p=0.9))
    assert error.value.status_code == 422
    assert state.phase is SessionPhase.READY
    assert state.task is None
    agent._run_sandbox_episode.assert_not_awaited()

    await agent.responses(request(state), NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert state.phase is SessionPhase.ACTIVATED
    agent._run_sandbox_episode.assert_awaited_once()


def test_http_close_retry_survives_other_session_closes(agent, state, monkeypatch):
    monkeypatch.setattr("responses_api_agents.hermes_agent.app.monotonic", lambda: 100.0)

    async def initialize(agent_session_id, body):
        return replace(state, request=body, close_lock=asyncio.Lock())

    agent._initialize_agent_session_state = AsyncMock(side_effect=initialize)
    with TestClient(agent.setup_webserver()) as client:

        def seed_and_close(index):
            client.cookies.clear()
            body = state.request.model_dump(mode="json")
            body["agent_session_id"] = f"session-{index}"
            body["episode_id"] = {"rollout_id": f"episode-{index}"}
            seed = client.post("/v1/agent_sessions", json=body)
            assert seed.status_code == 200
            cookies = dict(client.cookies)
            close = {"agent_session_id": seed.json()["agent_session_id"], "episode_id": body["episode_id"]}
            result = client.post("/v1/agent_sessions/close", json=close)
            assert result.status_code == 200
            return cookies, close, result.json()

        cookies, close, first = seed_and_close(0)
        # A's close succeeds, but its caller loses the reply while unrelated sessions finish.
        for index in range(1, 66):
            seed_and_close(index)
        client.cookies.clear()
        client.cookies.update(cookies)
        retry = client.post("/v1/agent_sessions/close", json=close)
        assert retry.status_code == 200
        assert retry.json() == first
    assert state.sandbox.disconnect.await_count == 66  # No second cleanup for A.


async def test_close_receipt_expires_without_extending_on_retry(agent, state, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("responses_api_agents.hermes_agent.app.monotonic", lambda: clock[0])
    agent.config.session_close_retry_window_seconds = 10
    agent._agent_sessions["session"] = state
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
    first = await agent.close_agent_session(request(state), close)
    clock[0] = 109.0
    assert await agent.close_agent_session(request(state), close) == first
    clock[0] = 110.0
    with pytest.raises(HTTPException) as error:
        await agent.close_agent_session(request(state), close)
    assert error.value.status_code == 409
    assert not agent._closed_agent_sessions
    agent._create_episode = AsyncMock(side_effect=AssertionError("host fallback"))
    with pytest.raises(HTTPException) as error:
        await agent.responses(request(state), NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert error.value.status_code == 409
    state.sandbox.disconnect.assert_awaited_once()


async def test_close_retry_window_starts_after_cleanup(agent, state, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("responses_api_agents.hermes_agent.app.monotonic", lambda: clock[0])
    agent.config.session_close_retry_window_seconds = 10
    agent._agent_sessions["session"] = state

    async def disconnect():
        clock[0] = 200.0  # Cleanup itself takes longer than the retry window.

    state.sandbox.disconnect.side_effect = disconnect
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
    first = await agent.close_agent_session(request(state), close)
    clock[0] = 209.0
    assert await agent.close_agent_session(request(state), close) == first
    state.sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("window", [0, -1, float("inf")])
def test_close_retry_window_must_be_positive_and_finite(agent, window):
    config = agent.config.model_dump() | {"session_close_retry_window_seconds": window}
    with pytest.raises(ValidationError, match="session_close_retry_window_seconds"):
        HermesAgentConfig.model_validate(config)


async def test_close_cancels_activation_and_rejects_duplicate(agent, state):
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def activate(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    agent._agent_sessions["session"] = state
    agent._run_sandbox_episode = AsyncMock(side_effect=activate)
    body = NeMoGymResponseCreateParamsNonStreaming(input="task")
    running = asyncio.create_task(agent.responses(request(state), body))
    await asyncio.wait_for(started.wait(), 5)
    assert state.phase is SessionPhase.ACTIVATED
    with pytest.raises(HTTPException) as error:
        await agent.responses(request(state), body)
    assert error.value.status_code == 409
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
    first, second = await asyncio.gather(
        agent.close_agent_session(request(state), close), agent.close_agent_session(request(state), close)
    )
    assert first == second
    assert stopped.is_set()
    assert state.phase is SessionPhase.CLOSING
    with pytest.raises(asyncio.CancelledError):
        await running
    state.sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("runner_started", [False, True], ids=["not-launched", "cleanup-confirmed"])
async def test_close_failure_keeps_session_for_retry(
    agent: HermesAgent, state: HermesAgentSessionState, runner_started: bool
) -> None:
    agent._agent_sessions["session"] = state
    if runner_started:
        state.runner_cleanup = RunnerCleanup.UNCONFIRMED
        agent._download_json = AsyncMock(return_value={"cleanup_confirmed": True})
    state.sandbox.exec.side_effect = [SimpleNamespace(return_code=1), SimpleNamespace(return_code=0)]
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
    with pytest.raises(RuntimeError, match="session files"):
        await agent.close_agent_session(request(state), close)
    assert agent._agent_sessions["session"] is state
    assert state.phase is SessionPhase.CLOSING
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
    agent._agent_sessions["session"] = state
    agent._download_json = AsyncMock(return_value=receipt)
    if receipt is None:
        agent._download_json.side_effect = FileNotFoundError("missing cleanup receipt")
    state.sandbox.exec.side_effect = AssertionError("Owned close must not depend on sandbox exec")
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)

    response = await agent.close_agent_session(request(state), close)
    assert response.agent_observations == state.observations
    assert state.runner_cleanup is RunnerCleanup.CONFIRMED
    assert "session" not in agent._agent_sessions
    assert await agent.close_agent_session(request(state), close) == response
    state.sandbox.stop.assert_awaited_once()
    state.sandbox.disconnect.assert_not_awaited()
    state.sandbox.exec.assert_not_awaited()
    agent._download_json.assert_not_awaited()


@pytest.mark.parametrize("failure", [RuntimeError, TimeoutError, asyncio.CancelledError])
async def test_owned_stop_failure_keeps_close_retryable(agent, state, failure):
    state.owns_sandbox = True
    state.runner_cleanup = RunnerCleanup.UNCONFIRMED
    agent._agent_sessions["session"] = state
    state.sandbox.stop.side_effect = [failure("stop failed"), None]
    agent._download_json = AsyncMock(side_effect=FileNotFoundError("missing cleanup receipt"))
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)

    with pytest.raises(failure, match="stop failed"):
        await agent.close_agent_session(request(state), close)
    assert agent._agent_sessions["session"] is state
    assert state.phase is SessionPhase.CLOSING
    assert state.runner_cleanup is RunnerCleanup.UNCONFIRMED
    assert "session" not in agent._closed_agent_sessions
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
    agent._agent_sessions["session"] = state
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
    wrapper = tmp_path / "delayed_runner.py"
    wrapper.write_text(
        "import pathlib,runpy,sys,time\n"
        f"sys.path.insert(0, {str(Path(hermes_app._SANDBOX_RUNNER).parent)!r})\n"
        f"pathlib.Path({str(ready)!r}).touch()\n"
        f"while not pathlib.Path({str(directory / 'runner.stop')!r}).exists(): time.sleep(0.01)\n"
        # Keep the interpreter in the pre-handler window while close sends TERM.
        "time.sleep(0.15)\n"
        f"runner=runpy.run_path({hermes_app._SANDBOX_RUNNER!r})\n"
        "def unexpected_worker(*args, **kwargs):\n"
        f"    pathlib.Path({str(worker_started)!r}).touch()\n"
        "    raise AssertionError('Worker must not start after the stop marker')\n"
        "runner['subprocess'].Popen=unexpected_worker\n"
        "raise SystemExit(runner['main']())\n"
    )
    monkeypatch.setattr(hermes_app, "_SANDBOX_RUNNER", str(wrapper))
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


@pytest.mark.parametrize("overrides", [{}, {"max_output_tokens": 32, "temperature": 0.0}])
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
    # Execute the real launch prefix: cleanup must receive the shell's PID, not a literal "$".
    launch_prefix, separator, _ = launch_command.partition(" && exec ")
    assert separator
    process = await asyncio.create_subprocess_exec("sh", "-c", launch_prefix)
    assert await process.wait() == 0
    assert int((tmp_path / "runner.pid").read_text()) == process.pid
    payload = agent._upload_json.await_args.args[2]
    assert payload["user_message"] == "Fix the bug"
    assert payload["model_enable_thinking"] is False
    assert payload["history"] == []
    assert payload["system_message"] == "Configured instruction\n\nRequest instruction"
    assert payload["max_tokens"] == overrides.get("max_output_tokens", 500)
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
def test_unsupported_requests_are_rejected(agent, override):
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": "task"} | override)
    with pytest.raises(HTTPException) as error:
        agent._validate_sandbox_request(body)
    assert error.value.status_code == 422


def test_text_history_and_system_message_are_preserved(agent):
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": [{"type": "input_text", "text": "First question"}]},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up"},
        ]
    )
    assert agent._validate_sandbox_request(body).input == body.input


async def test_required_resources_tools_are_rejected_before_connect(agent, state):
    from nemo_gym.tool_access import DirectHTTPToolAccess

    state.request.tool_accesses = [DirectHTTPToolAccess(name="required", required=True, base_url="http://tools")]
    with pytest.raises(HTTPException, match="only MCP tools"):
        await agent._initialize_agent_session_state("session", state.request)


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
    assert not agent._agent_sessions
    state.sandbox.disconnect.assert_awaited_once()


async def test_unknown_close_tombstone_and_locks_expire(agent, state, monkeypatch):
    agent.config.session_lifetime_seconds = 20
    clock = [100.0]
    monkeypatch.setattr("responses_api_agents.hermes_agent.app.monotonic", lambda: clock[0])
    agent.config.session_close_retry_window_seconds = 10
    close = AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
    stale_request = SimpleNamespace(session={})
    await agent.close_agent_session(stale_request, close)
    with pytest.raises(HTTPException, match="already closed"):
        await agent.seed_agent_session(SimpleNamespace(session={}), state.request)
    clock[0] = 110.0
    agent._expire_closed_agent_sessions()
    assert not agent._closed_agent_sessions
    with pytest.raises(HTTPException, match="already closed"):
        await agent.seed_agent_session(SimpleNamespace(session={}), state.request)
    with pytest.raises(HTTPException, match="expired"):
        await agent.close_agent_session(SimpleNamespace(session={}), close)
    clock[0] = 120.0
    agent._expire_closed_agent_sessions()
    assert not agent._closed_agent_session_ids
    assert not agent._agent_session_locks
    with pytest.raises(HTTPException, match="expired"):
        await agent.seed_agent_session(stale_request, state.request)
    with pytest.raises(HTTPException, match="expired"):
        await agent.close_agent_session(stale_request, close)


@pytest.mark.parametrize("marker", [None, "", 0, [], {}])
async def test_malformed_cookie_cannot_fall_back_or_seed(agent, state, marker):
    malformed = SimpleNamespace(session={"agent_session_id": marker})
    agent._create_response = AsyncMock(side_effect=AssertionError("host fallback"))
    with pytest.raises(HTTPException, match="Invalid Hermes"):
        await agent.responses(malformed, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    with pytest.raises(HTTPException, match="Invalid Hermes"):
        await agent.seed_agent_session(malformed, state.request)
    with pytest.raises(HTTPException, match="Invalid Hermes"):
        await agent.close_agent_session(
            malformed, AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id)
        )
    with pytest.raises(HTTPException, match="Invalid Hermes"):
        await agent.run(malformed, HermesAgentRunRequest(responses_create_params={"input": "task"}))


@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_abandoned_session_is_closed_or_retained_fail_closed(agent, state, cleanup_fails, caplog):
    agent.config.session_lifetime_seconds = 0.001
    agent._initialize_agent_session_state = AsyncMock(return_value=state)
    await agent.seed_agent_session(SimpleNamespace(session={}), state.request)
    if cleanup_fails:
        state.sandbox.disconnect.side_effect = RuntimeError("disconnect unavailable")
    reaper = agent._session_reapers["session"]
    await asyncio.wait_for(asyncio.shield(reaper), timeout=1)
    if cleanup_fails:
        assert agent._agent_sessions["session"].phase is SessionPhase.CLOSING
        assert "owner recovery required" in caplog.text
        with pytest.raises(HTTPException):
            await agent.responses(request(state), NeMoGymResponseCreateParamsNonStreaming(input="task"))
        state.sandbox.disconnect.side_effect = None
        await agent.close_agent_session(
            SimpleNamespace(session={}),
            AgentCloseSessionRequest(agent_session_id="session", episode_id=state.request.episode_id),
        )
    assert not agent._agent_sessions
    assert not agent._session_reapers
    assert "session" in agent._closed_agent_sessions
    state.sandbox.stop.assert_not_awaited()


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
        assert agent._agent_sessions["session"].phase is SessionPhase.CLOSING
        with pytest.raises(HTTPException, match="closing"):
            await agent.seed_agent_session(SimpleNamespace(session={}), body)
    else:
        assert not agent._agent_sessions
    sandbox.exec.side_effect = None
    cleanup.side_effect = None
    receipt = await agent.close_agent_session(
        SimpleNamespace(session={}),
        AgentCloseSessionRequest(agent_session_id="session", episode_id=body.episode_id),
    )
    assert receipt.agent_session_id == "session"
    assert not agent._agent_sessions
    assert not agent._session_reapers
    if owns_sandbox:
        sandbox.disconnect.assert_not_awaited()
    else:
        sandbox.stop.assert_not_awaited()
