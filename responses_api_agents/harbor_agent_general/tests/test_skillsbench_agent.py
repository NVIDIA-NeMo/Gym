# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral checks for the OpenHands adapter used by SkillsBench."""

import asyncio
import shutil
import subprocess
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from harbor.models.task.config import EnvironmentConfig
from openhands.tools.terminal.definition import TerminalAction

from benchmarks.skillsbench import openhands_agent
from benchmarks.skillsbench.openhands_agent import (
    _EXECUTORS,
    OpenSandboxTerminalExecutor,
    SkillsBenchOpenHands,
    SkillsBenchTerminalTool,
    _RunControl,
)
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.harbor_agent.tests.test_environment import _make_environment, _provider
from responses_api_agents.harbor_agent.tests.test_environment import _reset_fake_provider as _reset_fake_provider


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", ["timeout", "transport"])
async def test_provider_runtime_failure_deletes_sandbox_before_executor_returns(tmp_path, error_type):
    environment = _make_environment(tmp_path)
    await environment.start(force_build=False)
    _provider().queue_exec_result(
        SandboxExecResult(return_code=125, error_type=error_type, stdout=None, stderr="deadline")
    )
    executor = OpenSandboxTerminalExecutor(environment, asyncio.get_running_loop(), "/tmp", _RunControl())
    action = TerminalAction(command="sleep 2; touch /tmp/answer", timeout=1)
    if error_type == "timeout":
        with pytest.raises(RuntimeError, match="without confirmed termination"):
            await asyncio.to_thread(executor, action)
    else:
        observation = await asyncio.to_thread(executor, action)
        assert observation.is_error
    assert _provider().closed_handles == ["sbx-123"]
    assert not executor.control.active_commands


@pytest.mark.asyncio
@pytest.mark.parametrize("return_code", [124, 125])
async def test_terminal_process_exit_does_not_delete_sandbox(tmp_path, return_code):
    environment = _make_environment(tmp_path)
    await environment.start(force_build=False)
    try:
        _provider().queue_exec_result(
            SandboxExecResult(return_code=return_code, error_type=None, stdout=None, stderr=None)
        )
        executor = OpenSandboxTerminalExecutor(environment, asyncio.get_running_loop(), "/tmp", _RunControl())
        observation = await asyncio.to_thread(executor, TerminalAction(command=f"exit {return_code}"))
        assert observation.exit_code == return_code
        assert observation.is_error
        assert _provider().closed_handles == []
    finally:
        await environment.stop(delete=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("skill_mode", ["with-skill", "no-skill"])
async def test_setup_enforces_skill_condition_for_any_provider(skill_mode):
    environment = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=0)))
    await SkillsBenchOpenHands.setup(SimpleNamespace(skill_mode=skill_mode), environment)
    if skill_mode == "with-skill":
        environment.exec.assert_not_awaited()
    else:
        environment.exec.assert_awaited_once()
        command = environment.exec.call_args.args[0]
        assert command.startswith("rm -rf ")
        assert set(command.split()[2:]) == {
            f"/root/.{agent}/skills"
            for agent in ("agents", "codex", "claude", "gemini", "opencode", "goose", "factory", "openhands")
        } | {"/root/.opencode/skill"}


@pytest.mark.asyncio
async def test_no_skill_setup_rejects_failed_removal():
    environment = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=1, stderr="denied")))
    with pytest.raises(RuntimeError, match="denied"):
        await SkillsBenchOpenHands.setup(SimpleNamespace(skill_mode="no-skill"), environment)


@pytest.mark.asyncio
async def test_deadline_discards_completed_command_descendants_and_joins_worker():
    waiting_for_model = threading.Event()
    worker_finished = threading.Event()
    stopped = threading.Event()
    observations = []

    async def sandbox_exec(command, **_kwargs):
        assert not stopped.is_set()
        if command == "pwd":
            return SimpleNamespace(return_code=0, stdout="/workspace\n")
        return SimpleNamespace(return_code=0, stdout="service ready", stderr="")

    async def sandbox_stop(**_kwargs):
        stopped.set()

    def run_openhands(_instruction, _environment, _context, instance_id, _cwd, control):
        try:
            executor = _EXECUTORS[instance_id]
            observations.append(executor(TerminalAction(command="nohup server >/tmp/server.log 2>&1 &")))
            observations.append(executor(TerminalAction(command="curl localhost:8000")))
            waiting_for_model.set()
            assert control.cancelled.wait(timeout=5)
        finally:
            worker_finished.set()

    environment = SimpleNamespace(exec=sandbox_exec, stop=sandbox_stop)
    agent = SimpleNamespace(_run_openhands=run_openhands)
    task = asyncio.create_task(SkillsBenchOpenHands.run(agent, "Task", environment, None))
    assert await asyncio.to_thread(waiting_for_model.wait, 5)
    assert [observation.exit_code for observation in observations] == [0, 0]
    assert not stopped.is_set()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()
    assert worker_finished.is_set()


@pytest.mark.asyncio
async def test_deadline_does_not_signal_completed_command_with_stale_pid():
    control = _RunControl()
    future = Future()
    future.set_result(SimpleNamespace(return_code=0))
    # Completion can race with cleanup before the executor updates bookkeeping.
    control.active_commands["/tmp/skillsbench-openhands-stale.pid"] = future
    control.command_started = True
    environment = SimpleNamespace(exec=AsyncMock(), stop=AsyncMock())
    await SkillsBenchOpenHands._terminate_active_commands(environment, control)
    environment.stop.assert_awaited_once_with(delete=True)
    environment.exec.assert_not_awaited()


@pytest.mark.parametrize("overrides", [{}, {"temperature": 0, "top_p": 1, "max_output_tokens": 1024}])
def test_sampling_overrides_reach_openhands_llm_without_mutating_model_info(tmp_path, monkeypatch, overrides):
    model_info = {"max_input_tokens": 65536, "max_output_tokens": 32768}
    agent = SkillsBenchOpenHands(
        logs_dir=tmp_path,
        model_name="test-policy",
        api_base="http://model.example/v1",
        model_info=model_info,
        temperature=1.0,
        top_p=0.95,
        responses_create_params=overrides,
    )
    received = {}

    def capture_llm(**kwargs):
        received.update(kwargs)
        raise RuntimeError("stop after LLM construction")

    monkeypatch.setattr(openhands_agent, "LLM", capture_llm)
    with pytest.raises(RuntimeError, match="stop after LLM construction"):
        agent._run_openhands("task", None, None, "id", "/root", _RunControl())
    expected = {"temperature": 1.0, "top_p": 0.95, "max_output_tokens": 32768} | overrides
    assert all(received[key] == value for key, value in expected.items())
    assert received["max_input_tokens"] == 65536
    assert model_info == {"max_input_tokens": 65536, "max_output_tokens": 32768}


@pytest.mark.asyncio
async def test_timeout_waits_for_worker_before_verifier_can_start() -> None:
    started = threading.Event()
    finished = threading.Event()

    def run_openhands(_instruction, _environment, _context, _instance_id, _cwd, control):
        started.set()
        assert control.cancelled.wait(timeout=5)
        time.sleep(0.05)
        finished.set()

    agent = SimpleNamespace(_run_openhands=run_openhands)
    environment = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="/workspace\n")))
    task = asyncio.create_task(SkillsBenchOpenHands.run(agent, "Task", environment, None))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


@pytest.mark.asyncio
async def test_frozen_terminal_tool_describes_stateless_shell_and_keeps_executor() -> None:
    environment = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="regression\n", stderr=""))
    )
    executor = OpenSandboxTerminalExecutor(environment, asyncio.get_running_loop(), "/tmp", _RunControl())
    _EXECUTORS["test"] = executor
    try:
        state = SimpleNamespace(workspace=SimpleNamespace(working_dir="/tmp"))
        tool = SkillsBenchTerminalTool.create(state, "test")[0]
        assert tool.model_config["frozen"]
        assert "fresh shell" in tool.description
        assert "do not persist" in tool.description
        assert tool.executor is executor
        observation = await asyncio.to_thread(tool.executor, TerminalAction(command="echo regression"))
        assert observation.exit_code == 0
        environment.exec.assert_awaited_once()
        assert "echo regression" in environment.exec.call_args.args[0]
    finally:
        _EXECUTORS.pop("test")


@pytest.mark.asyncio
@pytest.mark.parametrize("detached", [False, True])
async def test_deadline_discards_active_command_and_joins_worker(detached):
    started = asyncio.Event()
    worker_finished = threading.Event()
    stopped = False
    exec_cancelled = False
    observed = []

    async def sandbox_exec(command, **_kwargs):
        nonlocal exec_cancelled
        if command == "pwd":
            return SimpleNamespace(return_code=0, stdout="/workspace\n")
        observed.append(command)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            # A client-side cancellation is safe only after sandbox disposal.
            assert stopped
            exec_cancelled = True

    async def sandbox_stop(**_kwargs):
        nonlocal stopped
        stopped = True

    def run_openhands(_instruction, _environment, _context, instance_id, _cwd, _control):
        command = "sleep 600; touch answer"
        if detached:
            command = "setsid bash -c 'sleep 10; touch answer' </dev/null >/tmp/bg.log 2>&1 & sleep 600"
        try:
            observation = _EXECUTORS[instance_id](TerminalAction(command=command, timeout=600))
            assert observation.is_error
        finally:
            worker_finished.set()

    environment = SimpleNamespace(exec=sandbox_exec, stop=sandbox_stop)
    agent = SimpleNamespace(_run_openhands=run_openhands)
    task = asyncio.create_task(SkillsBenchOpenHands.run(agent, "Task", environment, None))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped
    assert worker_finished.is_set()
    assert exec_cancelled
    # No PID inspection or group signal can prove detached descendants exited.
    assert len(observed) == 1


@pytest.mark.asyncio
async def test_failed_disposal_does_not_release_active_exec_as_if_terminated():
    control = _RunControl()
    control.command_started = True
    future = Future()
    control.active_commands["command"] = future
    environment = SimpleNamespace(exec=AsyncMock(), stop=AsyncMock(side_effect=RuntimeError("delete failed")))
    with pytest.raises(RuntimeError, match="delete failed"):
        await SkillsBenchOpenHands._terminate_active_commands(environment, control)
    assert not future.done()
    environment.exec.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_cancellation_discards_sandbox_before_returning():
    environment = SimpleNamespace(exec=AsyncMock(side_effect=asyncio.CancelledError), stop=AsyncMock())
    control = _RunControl()
    executor = OpenSandboxTerminalExecutor(environment, asyncio.get_running_loop(), "/tmp", control)
    observation = await asyncio.to_thread(executor, TerminalAction(command="sleep 600"))
    assert observation.is_error
    environment.stop.assert_awaited_once_with(delete=True)
    assert control.sandbox_discarded.is_set()


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker CLI is unavailable")
@pytest.mark.parametrize("completed", [False, True])
async def test_docker_deadline_discards_detached_processes(tmp_path, completed):
    available = subprocess.run(["docker", "info"], capture_output=True, timeout=15)
    if available.returncode:
        pytest.skip("Docker daemon is unavailable")
    environment = _make_environment(
        tmp_path,
        sandbox_provider={"docker": {}},
        task_env_config=EnvironmentConfig(docker_image="ubuntu:24.04", cpus=1, memory_mb=512),
    )
    finished = threading.Event()
    waiting = threading.Event()
    observations = []

    def run_openhands(_instruction, _environment, _context, instance_id, _cwd, control):
        command = (
            "setsid bash -c 'echo ready >/tmp/service-ready; sleep 3; touch /tmp/answer' "
            "</dev/null >/tmp/bg.log 2>&1 &"
        )
        if not completed:
            command += " sleep 600"
        try:
            observations.append(_EXECUTORS[instance_id](TerminalAction(command=command, timeout=600)))
            if completed:
                waiting.set()
                assert control.cancelled.wait(timeout=15)
        finally:
            finished.set()

    task = None
    try:
        await environment.start(force_build=False)
        task = asyncio.create_task(
            SkillsBenchOpenHands.run(SimpleNamespace(_run_openhands=run_openhands), "Task", environment, None)
        )
        for _ in range(20):
            probe = await environment.exec("test -f /tmp/service-ready && test ! -f /tmp/answer")
            if probe.return_code == 0:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("Background service was not observable before the write deadline")
        if completed:
            assert await asyncio.to_thread(waiting.wait, 5)
            assert observations[0].exit_code == 0
        else:
            assert not observations  # The first exec is still blocked on sleep.
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert environment._sandbox is None
        await asyncio.sleep(3.1)
        with pytest.raises(RuntimeError):
            await environment.exec("test -f /tmp/answer")
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await environment.stop(delete=True)
