# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral checks for the OpenHands adapter used by SkillsBench."""

import asyncio
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openhands.tools.terminal.definition import TerminalAction

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
@pytest.mark.parametrize("shell_noise", ["", "bash: no job control in this shell\n"])
async def test_timeout_terminates_active_sandbox_command_before_verification(shell_noise: str) -> None:
    started = asyncio.Event()
    terminated = asyncio.Event()
    worker_finished = threading.Event()
    late_write = False
    stopped = False

    async def sandbox_exec(command, **_kwargs):
        nonlocal late_write
        if command == "pwd":
            return SimpleNamespace(return_code=0, stdout="/workspace\n")
        if command.startswith("test -s "):
            return SimpleNamespace(return_code=0, stdout="")
        if command.startswith("cat /tmp/skillsbench-openhands-"):
            return SimpleNamespace(return_code=0, stdout=shell_noise + "12345\n")
        if command.startswith("pgid=$(cat /tmp/skillsbench-openhands-"):
            return SimpleNamespace(return_code=0, stdout=shell_noise + "\n__OPENHANDS_PGID__12345\n")
        if command.startswith("kill -TERM -- -"):
            # Interactive Bash ignores TERM and advances after its child dies.
            late_write = True
            terminated.set()
            return SimpleNamespace(return_code=0, stdout="")
        if command.startswith("kill -KILL -- -"):
            terminated.set()
            return SimpleNamespace(return_code=0, stdout="")
        assert command.startswith("setsid -w bash -c ")
        started.set()
        try:
            await asyncio.wait_for(terminated.wait(), timeout=5)
        except asyncio.TimeoutError:
            late_write = True
        return SimpleNamespace(return_code=143, stdout="", stderr="terminated")

    async def sandbox_stop(**_kwargs):
        nonlocal stopped
        stopped = True

    def run_openhands(_instruction, _environment, _context, instance_id, _cwd, _control):
        try:
            _EXECUTORS[instance_id](SimpleNamespace(command="sleep 600; touch answer", is_input=False, timeout=600))
        finally:
            worker_finished.set()

    environment = SimpleNamespace(exec=sandbox_exec, stop=sandbox_stop)
    agent = SimpleNamespace(_run_openhands=run_openhands)
    task = asyncio.create_task(SkillsBenchOpenHands.run(agent, "Task", environment, None))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert terminated.is_set()
    assert worker_finished.is_set()
    assert not late_write
    assert not stopped


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    ["12345\n", "__OPENHANDS_PGID__1\n", "__OPENHANDS_PGID__-42\n", "__OPENHANDS_PGID__42\n__OPENHANDS_PGID__43\n"],
)
async def test_timeout_refuses_unsafe_or_ambiguous_process_group(response: str) -> None:
    control = _RunControl()
    control.active_commands["/tmp/skillsbench-openhands-test.pid"] = Future()
    environment = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout=""),
                SimpleNamespace(return_code=0, stdout=response),
            ]
        )
    )
    with pytest.raises(RuntimeError, match="Invalid OpenHands process group"):
        await SkillsBenchOpenHands._terminate_active_commands(environment, control)
    assert environment.exec.await_count == 2
    assert all("kill " not in call.args[0] for call in environment.exec.await_args_list)
