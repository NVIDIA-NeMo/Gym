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


@pytest.mark.parametrize("alias", ["self-hosted-reasoning-model", "custom-checkpoint-alias"])
def test_self_hosted_alias_retains_reasoning_in_next_request(alias):
    from openhands.sdk.llm.message import Message, TextContent

    messages = [
        Message(role="user", content=[TextContent(text="Solve the task")]),
        Message(
            role="assistant",
            content=[TextContent(text="Next action")],
            reasoning_content="The computed intermediate value is 37.",
        ),
        Message(role="user", content=[TextContent(text="Continue")]),
    ]
    kwargs = dict(model=f"openai/{alias}", api_key="dummy", base_url="http://localhost:1/v1")
    original = openhands_agent.LLM(**kwargs)._to_chat_dicts(messages)
    preserved = openhands_agent.SkillsBenchLLM(**kwargs)._to_chat_dicts(messages)
    assert "reasoning_content" not in original[1]
    assert preserved[1].pop("reasoning_content") == "The computed intermediate value is 37."
    assert preserved == original
    assert messages[1].reasoning_content == "The computed intermediate value is 37."


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
    assert executor.control.cancelled.is_set()
    assert executor.control.sandbox_failure


@pytest.mark.asyncio
async def test_discarded_sandbox_stops_conversation_and_preserves_original_error(tmp_path):
    environment = _make_environment(tmp_path)
    await environment.start(force_build=False)
    _provider().queue_exec_result(SandboxExecResult(return_code=0, stdout="/tmp\n", stderr=None))
    _provider().queue_exec_result(
        SandboxExecResult(return_code=125, error_type="transport", stdout=None, stderr="provider disconnected")
    )
    from unittest.mock import Mock

    conversation = SimpleNamespace(pause=Mock())

    def run_openhands(_instruction, _environment, _context, instance_id, _cwd, control):
        control.conversation = conversation
        observation = _EXECUTORS[instance_id](TerminalAction(command="echo hello"))
        assert observation.is_error
        assert control.cancelled.is_set()

    agent = SimpleNamespace(_run_openhands=run_openhands)
    with pytest.raises(openhands_agent.SkillsBenchSandboxError, match="provider disconnected"):
        await SkillsBenchOpenHands.run(agent, "task", environment, None)
    assert _provider().closed_handles == ["sbx-123"]
    conversation.pause.assert_called_once()


def test_cancelled_run_prevents_additional_model_retries():
    control = _RunControl()
    control.on_llm_retry(1, 5, TimeoutError("temporary"))
    control.cancelled.set()
    with pytest.raises(openhands_agent._RunCancelled, match="do not start another model request"):
        control.on_llm_retry(2, 5, TimeoutError("temporary"))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancelled", "unrelated", "unrequested"])
async def test_sdk_wrapped_stop_preserves_deadline_and_unrelated_errors(tmp_path, monkeypatch, failure):
    from openhands.sdk.agent import base as sdk_agent_base
    from openhands.sdk.conversation.exceptions import ConversationRunError

    monkeypatch.setattr(sdk_agent_base, "has_vision_profile_available", lambda: False)
    started = threading.Event()
    finished = threading.Event()
    controls = []
    agent = SkillsBenchOpenHands(
        logs_dir=tmp_path,
        model_name="test-policy",
        api_base="http://unused.invalid/v1",
        skill_mode="no-skill",
    )
    run_openhands = agent._run_openhands

    def wrapped_run(*args):
        controls.append(args[-1])
        try:
            return run_openhands(*args)
        finally:
            finished.set()

    def step(_agent, _conversation, **_kwargs):
        control = controls[0]
        assert isinstance(_agent.condenser, openhands_agent.LLMSummarizingCondenser)
        assert _agent.condenser.llm is _agent.llm
        assert _agent.condenser.llm.retry_listener == control.on_llm_retry
        executor = next(value for value in _EXECUTORS.values() if value.control is control)
        executor(TerminalAction(command="echo ready"))
        started.set()
        if failure != "unrequested":
            assert control.cancelled.wait(timeout=5)
        if failure == "unrelated":
            raise ValueError("unrelated model failure")
        if failure == "unrequested":
            raise openhands_agent._RunCancelled("unexpected stop")
        control.on_llm_retry(1, 5, TimeoutError("temporary"))

    monkeypatch.setattr(agent, "_run_openhands", wrapped_run)
    # Exercise the real SDK exception wrapping, rather than mocking its wrapper.
    monkeypatch.setattr(openhands_agent.Agent, "step", step)
    environment = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="/workspace\n", stderr="")),
        stop=AsyncMock(),
    )
    context = openhands_agent.HarborAgentContext()
    task = asyncio.create_task(agent.run("Task", environment, context))
    assert await asyncio.to_thread(started.wait, 5)
    if failure != "unrequested":
        task.cancel()
    expected = asyncio.CancelledError if failure == "cancelled" else ConversationRunError
    with pytest.raises(expected):
        await task
    assert finished.is_set()
    assert not any(executor.control is controls[0] for executor in _EXECUTORS.values())
    if failure != "unrequested":
        environment.stop.assert_awaited_once_with(delete=True)
    else:
        environment.stop.assert_not_awaited()


@pytest.mark.parametrize("trigger", ["tokens", "provider_request"])
def test_native_condenser_reduces_history_and_preserves_task(tmp_path, monkeypatch, trigger):
    from openhands.sdk.context.condenser.utils import get_total_token_count
    from openhands.sdk.context.prompts import prompt
    from openhands.sdk.context.view import View
    from openhands.sdk.event import Condensation, MessageEvent, SystemPromptEvent
    from openhands.sdk.llm import Message, TextContent

    monkeypatch.setattr(prompt, "get_user_persistence_dir", lambda: tmp_path)
    llm = openhands_agent.LLM(
        model="openai/test-policy",
        base_url="http://unused.invalid/ng-rollout/3-1/v1",
        max_input_tokens=131072,
        max_output_tokens=32768,
        seed=1,
    )
    calls = []

    def token_count(_llm, messages, **_kwargs):
        # Deterministic token accounting; exercise the real SDK history selection
        # and summary application without a tokenizer download or model request.
        return sum(len(part.text) for message in messages for part in message.content)

    def generate(model, **kwargs):
        assert model is llm
        assert model.seed == 1
        assert model.base_url == "http://unused.invalid/ng-rollout/3-1/v1"
        assert model.max_output_tokens == 32768
        calls.append(kwargs)
        return SimpleNamespace(
            id="summary", message=Message(role="assistant", content=[TextContent(text="Progress.")])
        )

    monkeypatch.setattr(openhands_agent.LLM, "get_token_count", token_count)
    monkeypatch.setattr(openhands_agent.LLM, "generate", generate)
    condenser = openhands_agent.LLMSummarizingCondenser(llm=llm)
    prefix = [
        SystemPromptEvent(system_prompt=TextContent(text="Keep the task requirements."), tools=[]),
        MessageEvent(source="user", llm_message=Message(role="user", content=[TextContent(text="Original task.")])),
    ]
    short = View(events=prefix)
    assert condenser.condense(short, agent_llm=llm) is short
    assert not calls
    events = prefix + [
        MessageEvent(
            source="agent" if i % 2 == 0 else "user",
            llm_message=Message(
                role="assistant" if i % 2 == 0 else "user",
                content=[TextContent(text="progress " * (3000 if trigger == "tokens" else 10))],
            ),
        )
        for i in range(10)
    ]
    view = View(events=events, unhandled_condensation_request=trigger == "provider_request")
    result = condenser.condense(view, agent_llm=llm)
    assert isinstance(result, Condensation)
    compacted = result.apply(events)
    assert compacted[:2] == prefix
    assert len(compacted) < len(events)
    assert get_total_token_count(compacted, llm) < llm.max_input_tokens
    assert len(calls) == 1
    assert view.events == events  # Compaction does not alter the retained trace.


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
        assert f"Command finished with exit code {return_code}" in observation.to_llm_content[-1].text
        assert _provider().closed_handles == []
    finally:
        await environment.stop(delete=True)


@pytest.mark.asyncio
async def test_terminal_model_feedback_reports_completion_and_updated_directory(tmp_path):
    environment = _make_environment(tmp_path)
    await environment.start(force_build=False)
    try:

        async def exec_command(command, **kwargs):
            assert kwargs["cwd"] == "/"
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                cwd=kwargs["cwd"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            return SimpleNamespace(return_code=process.returncode, stdout=stdout.decode(), stderr=stderr.decode())

        environment.exec = exec_command
        executor = OpenSandboxTerminalExecutor(environment, asyncio.get_running_loop(), "/tmp", _RunControl())
        observation = await asyncio.to_thread(executor, TerminalAction(command=f"cd '{tmp_path}'; echo done"))
        content = "\n".join(part.text for part in observation.to_llm_content)
        assert "[Command finished with exit code 0]" in content
        assert f"[Current working directory: {tmp_path}]" in content
        assert "__OPENHANDS_CWD_" not in content
        assert executor.cwd == str(tmp_path)
        assert not observation.is_error
    finally:
        await environment.stop(delete=True)


@pytest.mark.asyncio
async def test_unsupported_interactive_input_has_completed_error_feedback():
    environment = SimpleNamespace(exec=AsyncMock())
    executor = OpenSandboxTerminalExecutor(environment, asyncio.get_running_loop(), "/tmp", _RunControl())
    observation = executor(TerminalAction(command="C-c", is_input=True))
    assert observation.is_error
    assert "[Command finished with exit code 1]" in observation.to_llm_content[-1].text
    environment.exec.assert_not_called()


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


@pytest.mark.parametrize("llm_timeout_seconds", [300, 1800])
@pytest.mark.parametrize("overrides", [{}, {"temperature": 0, "top_p": 1, "max_output_tokens": 131072, "seed": 0}])
def test_sampling_overrides_reach_openhands_llm_without_mutating_model_info(
    tmp_path, monkeypatch, overrides, llm_timeout_seconds
):
    model_info = {"max_input_tokens": 65536, "max_output_tokens": 32768}
    agent = SkillsBenchOpenHands(
        logs_dir=tmp_path,
        model_name="test-policy",
        api_base="http://model.example/v1",
        model_info=model_info,
        temperature=1.0,
        top_p=0.95,
        responses_create_params=overrides,
        llm_timeout_seconds=llm_timeout_seconds,
    )
    received = {}

    def capture_llm(**kwargs):
        received.update(kwargs)
        raise RuntimeError("stop after LLM construction")

    monkeypatch.setattr(openhands_agent, "SkillsBenchLLM", capture_llm)
    with pytest.raises(RuntimeError, match="stop after LLM construction"):
        agent._run_openhands("task", None, None, "id", "/root", _RunControl())
    expected = {"temperature": 1.0, "top_p": 0.95, "max_output_tokens": 32768} | overrides
    assert all(received[key] == value for key, value in expected.items())
    assert received["max_input_tokens"] == 65536
    assert received["timeout"] == llm_timeout_seconds
    assert received["seed"] == overrides.get("seed")
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
        assert observation.command == "echo regression"
        assert "regression" in observation.text
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


@pytest.mark.asyncio
async def test_deleted_cwd_is_tool_error_and_next_action_can_recover(tmp_path):
    from harbor.environments.base import ExecResult

    class ShellEnvironment:
        stop = AsyncMock()

        async def exec(self, command, *, cwd=None, timeout_sec=None):
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
            return ExecResult(return_code=process.returncode, stdout=stdout.decode(), stderr=stderr.decode())

    # Quotes/spaces also exercise command escaping at the shell boundary.
    missing = tmp_path / "removed ' work"
    missing.mkdir()
    missing.rmdir()
    environment = ShellEnvironment()
    control = _RunControl()
    executor = OpenSandboxTerminalExecutor(environment, asyncio.get_running_loop(), str(missing), control)
    marker = tmp_path / "must-not-run"
    failed = await asyncio.to_thread(executor, TerminalAction(command=f"touch {marker}"))
    assert failed.is_error
    assert not marker.exists()
    assert executor.cwd == "/"
    import shlex

    recovered = await asyncio.to_thread(executor, TerminalAction(command=f"cd {shlex.quote(str(tmp_path))}; pwd"))
    assert not recovered.is_error
    assert executor.cwd == str(tmp_path)
    assert not control.cancelled.is_set()
    assert control.sandbox_failure is None
    environment.stop.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["provider_oom_killed", "provider_unavailable"])
async def test_resource_failure_keeps_typed_cause_and_disposes_before_return(kind):
    from nemo_gym.core import NgFailureEvent

    error = NgFailureEvent(boundary="sandbox", failure_kind=kind, failure_reason="control plane evidence")
    environment = SimpleNamespace(exec=AsyncMock(side_effect=error), stop=AsyncMock())
    control = _RunControl()
    executor = OpenSandboxTerminalExecutor(environment, asyncio.get_running_loop(), "/tmp", control)
    result = await asyncio.to_thread(executor, TerminalAction(command="work"))
    assert result.is_error
    environment.stop.assert_awaited_once_with(delete=True)
    expected = (
        openhands_agent.SkillsBenchResourceUnattributedError
        if kind == "provider_oom_killed"
        else openhands_agent.SkillsBenchSandboxError
    )
    assert control.sandbox_failure_type is expected
    assert control.cancelled.is_set()


@pytest.mark.asyncio
async def test_unknown_command_exit_discards_sandbox_before_return():
    environment = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=-1, stdout="", stderr="")), stop=AsyncMock()
    )
    control = _RunControl()
    executor = OpenSandboxTerminalExecutor(environment, asyncio.get_running_loop(), "/tmp", control)
    result = await asyncio.to_thread(executor, TerminalAction(command="work"))
    assert result.is_error
    environment.stop.assert_awaited_once_with(delete=True)
    assert control.cancelled.is_set()
    assert "outcome is unknown" in control.sandbox_failure


@pytest.mark.parametrize(
    "settings,expected", [({}, (3600, 5)), ({"llm_timeout_s": 900, "llm_num_retries": 2}, (900, 2))]
)
def test_the_llm_request_timeout_and_retries_reach_openhands(tmp_path, monkeypatch, settings, expected):
    agent = SkillsBenchOpenHands(
        logs_dir=tmp_path, model_name="test-policy", api_base="http://model.example/v1", **settings
    )
    received = {}

    def capture_llm(**kwargs):
        received.update(kwargs)
        raise RuntimeError("stop after LLM construction")

    monkeypatch.setattr(openhands_agent, "SkillsBenchLLM", capture_llm)
    with pytest.raises(RuntimeError, match="stop after LLM construction"):
        agent._run_openhands("task", None, None, "id", "/root", _RunControl())
    # OpenHands' own default is a 300 s request timeout, shorter than a slow but healthy policy needs.
    assert (received["timeout"], received["num_retries"]) == expected


@pytest.mark.asyncio
async def test_terminal_process_cleanup_does_not_match_command_transport(tmp_path):
    import shlex
    import uuid

    from harbor.environments.base import ExecResult

    class ShellEnvironment:
        stop = AsyncMock()

        async def exec(self, command, *, cwd=None, timeout_sec=None):
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
            return ExecResult(return_code=process.returncode, stdout=stdout.decode(), stderr=stderr.decode())

    # A unique process name confines pkill to this test's service and any shell
    # that incorrectly exposes the action text in its command-line arguments.
    name = f"skillsbench-service-{uuid.uuid4().hex}"
    service = await asyncio.create_subprocess_exec(
        "bash",
        "-c",
        'exec -a "$1" sleep 120',
        "bash",
        name,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    environment = ShellEnvironment()
    executor = OpenSandboxTerminalExecutor(environment, asyncio.get_running_loop(), str(tmp_path), _RunControl())
    try:
        assert service.returncode is None
        result = await asyncio.to_thread(
            executor, TerminalAction(command=f"pkill -f {shlex.quote(name)}; echo cleanup-finished", timeout=5)
        )
        await asyncio.wait_for(service.wait(), timeout=5)
        assert service.returncode != 0
        assert result.exit_code == result.metadata.exit_code == 0
        assert "cleanup-finished" in result.text
        assert executor.cwd == str(tmp_path)
        assert executor.control.sandbox_failure is None
        environment.stop.assert_not_awaited()
    finally:
        if service.returncode is None:
            service.kill()
            await service.wait()


@pytest.mark.asyncio
async def test_terminal_soft_wait_preserves_real_command_and_returns_its_result(tmp_path):
    import shlex

    from harbor.environments.base import ExecResult

    class ShellEnvironment:
        stop = AsyncMock()

        async def exec(self, command, *, cwd=None, timeout_sec=None):
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
            return ExecResult(return_code=process.returncode, stdout=stdout.decode(), stderr=stderr.decode())

    environment = ShellEnvironment()
    executor = OpenSandboxTerminalExecutor(environment, asyncio.get_running_loop(), str(tmp_path), _RunControl())
    marker = tmp_path / "finished"
    command = f"sleep 0.1; printf result > {shlex.quote(str(marker))}; cd /; echo complete"
    first = await asyncio.to_thread(executor, TerminalAction(command=command, timeout=0))
    assert first.exit_code == first.metadata.exit_code == -1
    assert not first.is_error
    assert not executor.control.cancelled.is_set()
    pending = executor.pending
    rejected = await asyncio.to_thread(executor, TerminalAction(command="touch must-not-run", timeout=0))
    assert rejected.is_error and rejected.exit_code == -1
    assert executor.pending is pending
    completed = await asyncio.to_thread(executor, TerminalAction(command="", timeout=3))
    assert completed.exit_code == completed.metadata.exit_code == 0
    assert completed.command == command
    assert "complete" in completed.text
    assert marker.read_text() == "result"
    assert not (tmp_path / "must-not-run").exists()
    assert executor.cwd == "/"
    assert executor.pending is None
    assert not executor.control.active_commands
    environment.stop.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_finished_conversation_drains_soft_wait_before_verification(cancel):
    started = asyncio.Event()
    release = asyncio.Event()
    exec_done = asyncio.Event()
    stopped = False

    async def sandbox_exec(command, **_kwargs):
        if command == "pwd":
            return SimpleNamespace(return_code=0, stdout="/tmp\n", stderr="")
        started.set()
        try:
            await release.wait()
            return SimpleNamespace(return_code=0, stdout="complete", stderr="")
        except asyncio.CancelledError:
            assert stopped
            raise
        finally:
            exec_done.set()

    async def sandbox_stop(**_kwargs):
        nonlocal stopped
        stopped = True

    def run_openhands(_instruction, _environment, _context, instance_id, _cwd, _control):
        result = _EXECUTORS[instance_id](TerminalAction(command="long work", timeout=0))
        assert result.exit_code == -1

    environment = SimpleNamespace(exec=sandbox_exec, stop=sandbox_stop)
    agent = SimpleNamespace(_run_openhands=run_openhands)
    task = asyncio.create_task(SkillsBenchOpenHands.run(agent, "Task", environment, None))
    await asyncio.wait_for(started.wait(), timeout=3)
    await asyncio.sleep(0.05)
    assert not task.done()
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)
        assert stopped
    else:
        release.set()
        await asyncio.wait_for(task, timeout=3)
        assert not stopped
    assert exec_done.is_set()
