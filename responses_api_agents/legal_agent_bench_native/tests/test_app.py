# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.agents import AgentSeedRejectedError, AgentSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, TaskId
from resources_servers.legal_agent_bench.lab_harbor.completion import CompletionState
from responses_api_agents.legal_agent_bench_native.app import (
    LabNativeSession,
    LegalAgentBenchNativeAgent,
    LegalAgentBenchNativeAgentConfig,
)


def seed(**kwargs):
    return AgentSeedSessionRequest(
        agent_session_id="a",
        episode_id=EpisodeId(rollout_id="r"),
        task_id=TaskId(taskset="lab", task_id="t"),
        **kwargs,
    )


def response(*, output=None, status="completed", usage=None):
    return NeMoGymResponse(
        id="resp",
        created_at=0,
        model="test",
        object="response",
        output=output or [],
        status=status,
        tools=[],
        tool_choice="auto",
        parallel_tool_calls=False,
        usage=usage,
    )


def agent():
    return LegalAgentBenchNativeAgent(
        config=LegalAgentBenchNativeAgentConfig(
            name="agent",
            entrypoint="app.py",
            host="localhost",
            port=0,
            model_server={"type": "responses_api_models", "name": "policy"},
            max_turns=2,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


@pytest.mark.asyncio
async def test_unseeded_sandbox_is_rejected():
    with pytest.raises(AgentSeedRejectedError, match="sandbox owned by the Harbor Environment Server"):
        await agent()._seed_agent_session_state(seed())


class Reply:
    cookies = {}

    def __init__(self, body):
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.mark.parametrize("tool", ["write", "bash"])
async def test_native_staging_preserves_literal_arguments(tmp_path, monkeypatch, tool):
    from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult
    from resources_servers.legal_agent_bench.prepare import TOOL_RUNNER_SOURCE
    from responses_api_agents.legal_agent_bench_native import app

    # Run the actual staging callback, upload helper, shell command, and document
    # tool. Only the provider and process supervisor are replaced for this test.
    tool_runner = tmp_path / "tool runner's copy.py"
    shutil.copyfile(TOOL_RUNNER_SOURCE, tool_runner)
    monkeypatch.setattr(app, "_TOOL_RUNNER", str(tool_runner))
    monkeypatch.setattr(app, "_TOOL_ENV", {key: str(tmp_path) for key in app._TOOL_ENV})
    sentinel = tmp_path / "unexpected-expansion"
    content = f"Single ' and double \" quotes\n$(touch {sentinel})\n`touch {sentinel}`\n* [abc] café\n"
    output = tmp_path / "answer's [draft] *.txt"
    command = (
        f"cat > {shlex.quote(str(output))} <<'LAB_LITERAL_CONTENT'\n"
        f"{content}LAB_LITERAL_CONTENT\nprintf 'script completed\\n'\n"
    )
    arguments = {"command": command} if tool == "bash" else {"file_path": str(output), "content": content}
    uploaded = {}
    activation_dirs = []

    async def run_process(argv):
        result = await asyncio.to_thread(
            subprocess.run,
            argv,
            cwd=tmp_path,
            env={**os.environ, "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}"},
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    async def execute(command, **kwargs):
        return SandboxExecResult(stdout=await run_process(["/bin/bash", "-c", command]), stderr="", return_code=0)

    async def upload(source, destination):
        uploaded[Path(destination).name] = Path(source).read_bytes()
        shutil.copyfile(source, destination)

    async def begin_activation(*, session_dir, **kwargs):
        activation_dirs.append(Path(session_dir))

    async def execute_activation(*, stage_activation, **kwargs):
        staged = await stage_activation()
        return await run_process(staged.argv)

    box = AsyncMock(spec=AsyncSandbox)
    box.exec.side_effect = execute
    box.upload.side_effect = upload
    session = SimpleNamespace(
        sandbox=box,
        begin_activation=begin_activation,
        execute=execute_activation,
        read_output_log=AsyncMock(),
    )
    state = LabNativeSession(request=seed(), session=session, completion=CompletionState())
    try:
        result = await agent()._execute_tool(
            state, tool, json.dumps(arguments), deadline=asyncio.get_running_loop().time() + 30
        )
        assert output.read_text() == content
        assert not sentinel.exists(), "Tool data was interpreted as shell commands"
        if tool == "bash":
            assert uploaded == {"command.sh": command.encode()}
            assert result == "script completed\n"
        else:
            assert set(uploaded) == {"arguments.json"}
            assert json.loads(uploaded["arguments.json"]) == arguments
            assert result == f"Wrote {len(content)} bytes to {output}"
    finally:
        for directory in activation_dirs:
            shutil.rmtree(directory, ignore_errors=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_missing", [False, True])
async def test_finish_checks_files_and_stops_later_tools(tmp_path, monkeypatch, repair_missing):
    from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult
    from resources_servers.legal_agent_bench.prepare import TOOL_RUNNER_SOURCE
    from responses_api_agents.legal_agent_bench_native import app

    workspace = tmp_path / "workspace"
    output = workspace / "output"
    output.mkdir(parents=True)
    monkeypatch.setattr(app, "_TOOL_RUNNER", str(TOOL_RUNNER_SOURCE))
    monkeypatch.setattr(
        app, "_TOOL_ENV", {**app._TOOL_ENV, "WORKSPACE_DIR": str(workspace), "OUTPUT_DIR": str(output)}
    )
    activation_dirs = []

    async def run_process(argv):
        result = await asyncio.to_thread(
            subprocess.run,
            argv,
            cwd=workspace,
            env={**os.environ, "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}"},
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    async def execute(command, **kwargs):
        return SandboxExecResult(stdout=await run_process(["/bin/bash", "-c", command]), stderr="", return_code=0)

    async def upload(source, destination):
        shutil.copyfile(source, destination)

    async def begin_activation(*, session_dir, **kwargs):
        activation_dirs.append(Path(session_dir))

    async def execute_activation(*, stage_activation, **kwargs):
        staged = await stage_activation()
        return await run_process(staged.argv)

    box = AsyncMock(spec=AsyncSandbox)
    box.exec.side_effect = execute
    box.upload.side_effect = upload
    session = SimpleNamespace(
        sandbox=box,
        expired=False,
        begin_activation=begin_activation,
        execute=execute_activation,
        read_output_log=AsyncMock(),
    )

    def call(call_id, name, **arguments):
        return {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(arguments)}

    calls = [
        call("write", "write", file_path="memo.txt", content="Original deliverable"),
        call("finish", "finish", summary="Completed memo", deliverables=["memo.txt"]),
        call("late-write", "write", file_path="memo.txt", content="Unexpected overwrite"),
    ]
    replies = [response(output=calls)]
    if repair_missing:
        replies.insert(0, response(output=[call("early-finish", "finish", summary="Done", deliverables=["memo.txt"])]))
    runner = agent()
    runner.server_client.post = AsyncMock(side_effect=[Reply(reply.model_dump(mode="json")) for reply in replies])
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())
    monkeypatch.setattr(app, "get_response_json", AsyncMock(side_effect=lambda reply: reply.body))
    state = LabNativeSession(request=seed(), session=session, completion=CompletionState())
    try:
        result = await runner._run_loop(state, NeMoGymResponseCreateParamsNonStreaming(input=[]), "/v1/responses", {})
        assert result.status == "completed"
        assert result.metadata["lab_finish_reason"] == "finish_tool"
        assert result.metadata["lab_finish_summary"] == "Completed memo"
        assert (output / "memo.txt").read_text() == "Original deliverable"
        assert runner.server_client.post.await_count == len(replies)
        assert state.tool_index == 2 + int(repair_missing)
        assert state.completion.rejections == int(repair_missing)
        observations = {item.call_id: item.output for item in result.output if item.type == "function_call_output"}
        assert observations["finish"] == "Finished."
        assert observations["late-write"] == "Task already finished; no further tools were executed."
        if repair_missing:
            assert observations["early-finish"].startswith("Not finished:")
            assert "memo.txt" in observations["early-finish"]
    finally:
        for directory in activation_dirs:
            shutil.rmtree(directory, ignore_errors=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("max_tokens", [None, 8192])
async def test_loop_retains_tools_usage_and_supplied_parameters(monkeypatch, max_tokens):
    from responses_api_agents.legal_agent_bench_native import app

    runner = agent()
    call = {
        "type": "function_call",
        "call_id": "call1",
        "name": "write",
        "arguments": '{"path":"answer.txt","content":"work"}',
    }
    usage = {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": None, "cache_write_tokens": 4},
        "output_tokens_details": {"reasoning_tokens": None},
    }
    replies = [response(output=[call], usage=usage), response(usage=usage)]
    runner.server_client.post = AsyncMock(side_effect=[Reply(reply.model_dump(mode="json")) for reply in replies])
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())
    monkeypatch.setattr(app, "get_response_json", AsyncMock(side_effect=lambda reply: reply.body))
    monkeypatch.setattr(LegalAgentBenchNativeAgent, "_execute_tool", AsyncMock(return_value="written"))
    state = LabNativeSession(request=seed(), session=SimpleNamespace(expired=False), completion=CompletionState())
    result = await runner._run_loop(
        state,
        NeMoGymResponseCreateParamsNonStreaming(
            input=[], top_p=None, parallel_tool_calls=False, max_output_tokens=max_tokens
        ),
        "/rollouts/r/v1/responses",
        {"session_id": "test-session"},
    )
    assert result.status == "completed" and result.usage.total_tokens == 30
    assert result.usage.input_tokens_details.cached_tokens is None
    assert result.usage.input_tokens_details.cache_write_tokens == 8
    assert result.output[1].type == "function_call_output" and result.output[1].output == "written"
    for request in runner.server_client.post.call_args_list:
        assert request.kwargs["server_name"] == "policy"
        assert request.kwargs["url_path"] == "/rollouts/r/v1/responses"
        assert request.kwargs["cookies"] == {"session_id": "test-session"}
    sent = runner.server_client.post.call_args_list[0].kwargs["json"]
    assert sent["top_p"] is None and sent["parallel_tool_calls"] is False
    assert sent["max_output_tokens"] == (max_tokens if max_tokens is not None else 65536)
    assert sent["input"][0]["role"] == "developer"


@pytest.mark.asyncio
async def test_max_turn_exhaustion_is_incomplete_and_model_failure_is_ungraded(monkeypatch):
    from responses_api_agents.legal_agent_bench_native import app

    runner = agent()
    runner.config.max_turns = 1
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())
    call = {"type": "function_call", "call_id": "call1", "name": "bash", "arguments": '{"command":"true"}'}
    monkeypatch.setattr(
        app, "get_response_json", AsyncMock(return_value=response(output=[call]).model_dump(mode="json"))
    )
    runner.server_client.post = AsyncMock(return_value=Reply({}))
    monkeypatch.setattr(LegalAgentBenchNativeAgent, "_execute_tool", AsyncMock(return_value="done"))
    state = LabNativeSession(request=seed(), session=SimpleNamespace(expired=False), completion=CompletionState())
    result = await runner._run_loop(state, NeMoGymResponseCreateParamsNonStreaming(input=[]), "/v1/responses", {})
    assert result.status == "incomplete" and result.metadata["lab_finish_reason"] == "max_turns_exceeded"
    monkeypatch.setattr(
        app, "get_response_json", AsyncMock(return_value=response(status="failed").model_dump(mode="json"))
    )
    result = await runner._run_loop(state, NeMoGymResponseCreateParamsNonStreaming(input=[]), "/v1/responses", {})
    assert result.status == "failed" and json.loads(result.metadata["ng_agent_outcome"])["failure_kind"] == "unknown"


@pytest.mark.asyncio
async def test_close_cancels_running_loop_and_disconnects_borrower():
    from nemo_gym.agents import SandboxSession
    from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult

    runner = agent()

    async def wait():
        await asyncio.Event().wait()

    task = asyncio.create_task(wait())
    sandbox = AsyncMock(spec=AsyncSandbox)
    sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
    box_session = SandboxSession(
        sandbox=sandbox, session_dir="/tmp/lab-session", workdir="/workspace", harness="lab", owns_sandbox=False
    )
    state = LabNativeSession(request=seed(), session=box_session, completion=CompletionState(), task=task)
    result = await runner._close_agent_session_state(state)
    assert task.cancelled()
    assert result.cleanup_confirmed and box_session.closed
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_activation_joins_one_loop_and_conflicting_body_is_rejected(monkeypatch):
    from fastapi import HTTPException, Request

    runner = agent()
    state = LabNativeSession(request=seed(), session=SimpleNamespace(closing=False), completion=CompletionState())
    monkeypatch.setattr(LegalAgentBenchNativeAgent, "_require_agent_session", lambda self, session_id: state)
    loop = AsyncMock(return_value=response())
    monkeypatch.setattr(LegalAgentBenchNativeAgent, "_run_loop", loop)
    request = Request(
        {
            "type": "http",
            "session": {"agent_session_id": "a"},
            "path_params": {"rollout_id": "r"},
            "headers": [],
            "path": "/rollouts/r/v1/responses",
        }
    )
    body = NeMoGymResponseCreateParamsNonStreaming(input=[])
    first, second = await asyncio.gather(runner.responses(request, body), runner.responses(request, body))
    assert first == second and loop.await_count == 1
    with pytest.raises(HTTPException) as error:
        await runner.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="another task"))
    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_explicit_context_rejection_is_scoreable_but_other_model_errors_propagate(monkeypatch):
    from aiohttp import ClientResponseError

    from responses_api_agents.legal_agent_bench_native import app

    runner = agent()
    runner.server_client.post = AsyncMock(return_value=Reply({}))
    error = ClientResponseError(request_info=SimpleNamespace(real_url="http://policy.test"), history=(), status=400)
    error.response_content = b'{"error":{"code":"context_length_exceeded"}}'
    monkeypatch.setattr(app, "raise_for_status", AsyncMock(side_effect=error))
    state = LabNativeSession(request=seed(), session=SimpleNamespace(expired=False), completion=CompletionState())
    result = await runner._run_loop(state, NeMoGymResponseCreateParamsNonStreaming(input=[]), "/v1/responses", {})
    assert result.status == "incomplete" and result.metadata["lab_context_limit"] == "true"
    error.response_content = b'{"error":{"code":"invalid_api_key"}}'
    with pytest.raises(ClientResponseError):
        await runner._run_loop(state, NeMoGymResponseCreateParamsNonStreaming(input=[]), "/v1/responses", {})


@pytest.mark.asyncio
async def test_native_operator_stop_is_explicitly_unsupported():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        await agent().stop_rollout("r")
    assert error.value.status_code == 501


@pytest.mark.asyncio
async def test_overall_agent_budget_cancels_model_wait_and_returns_scoreable_incomplete():
    runner = agent()
    runner.config.agent_timeout_seconds = 0.02
    canceled = asyncio.Event()

    async def post(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            canceled.set()

    runner.server_client.post = post
    state = LabNativeSession(request=seed(), session=SimpleNamespace(expired=False), completion=CompletionState())
    result = await runner._run_loop(state, NeMoGymResponseCreateParamsNonStreaming(input=[]), "/v1/responses", {})
    assert canceled.is_set() and result.status == "incomplete"
    assert result.metadata["lab_finish_reason"] == "agent_timeout" and result.usage is None


def _tool_loop(monkeypatch, *, budget=0.2):
    from nemo_gym.agents import SandboxSession
    from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult
    from responses_api_agents.legal_agent_bench_native import app

    runner = agent()
    runner.config.agent_timeout_seconds = budget
    runner.config.session_close_timeout_seconds = 0.5
    box = AsyncMock(spec=AsyncSandbox)
    box.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
    session = SandboxSession(
        sandbox=box, session_dir="/initial", workdir="/workspace", harness="lab", owns_sandbox=False
    )
    state = LabNativeSession(request=seed(), session=session, completion=CompletionState())
    call = {"type": "function_call", "call_id": "tool1", "name": "bash", "arguments": '{"command":"true"}'}
    runner.server_client.post = AsyncMock(return_value=Reply(response(output=[call]).model_dump(mode="json")))
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())
    monkeypatch.setattr(app, "get_response_json", AsyncMock(side_effect=lambda reply: reply.body))
    return runner, state, box


@pytest.mark.parametrize("phase", ["mkdir", "upload"])
@pytest.mark.parametrize("cleanup_fails", [False, True], ids=["clean", "cleanup-error"])
async def test_staging_deadline_uses_real_executor_and_requires_cleanup(monkeypatch, phase, cleanup_fails):
    from nemo_gym.agents import OutcomeAction, agent_outcome_policy
    from nemo_gym.sandbox import SandboxExecResult

    runner, state, box = _tool_loop(monkeypatch)
    cancelled = asyncio.Event()

    async def stall(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def execute(command, **kwargs):
        if command.startswith("mkdir") and phase == "mkdir":
            await stall()
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    box.exec.side_effect = execute
    if phase == "upload":
        box.upload.side_effect = stall
    if cleanup_fails:
        box.disconnect.side_effect = RuntimeError("provider disconnect failed")
    running = runner._run_loop(state, NeMoGymResponseCreateParamsNonStreaming(input=[]), "/v1/responses", {})
    if cleanup_fails:
        with pytest.raises(RuntimeError, match="session cleanup failed"):
            await asyncio.wait_for(running, 2)
        assert not state.session.closed
        box.disconnect.side_effect = None
        await state.session.close(timeout=0.5)
    else:
        result = await asyncio.wait_for(running, 2)
        assert result.status == "incomplete" and result.metadata["lab_finish_reason"] == "agent_timeout"
        assert agent_outcome_policy(result) == OutcomeAction.VERIFY
        assert result.output[-1].type == "function_call_output"
        assert result.output[-1].call_id == "tool1"
    assert cancelled.is_set() and state.session._stage_task.done()
    assert not state.session.launch_started and state.session.cleanup_confirmed and state.session.closed
    runner.server_client.post.assert_awaited_once()
    box.stop.assert_not_awaited()  # The resources server still owns the task sandbox.


async def test_deadline_during_rearm_waits_for_previous_cleanup_and_never_stages(monkeypatch):
    runner, state, box = _tool_loop(monkeypatch)
    cancelling = asyncio.Event()
    release = asyncio.Event()

    async def previous_stage():
        try:
            await asyncio.Event().wait()
        finally:
            cancelling.set()
            await release.wait()

    state.session._stage_task = asyncio.create_task(previous_stage())
    await asyncio.sleep(0)
    running = asyncio.create_task(
        runner._run_loop(state, NeMoGymResponseCreateParamsNonStreaming(input=[]), "/v1/responses", {})
    )
    try:
        await asyncio.wait_for(cancelling.wait(), 1)
        await asyncio.sleep(0.3)  # Exceed the agent budget while cleanup is still pending.
        assert not running.done()
        release.set()
        result = await asyncio.wait_for(running, 1)
        assert result.status == "incomplete" and result.metadata["lab_finish_reason"] == "agent_timeout"
        assert not state.session.launch_started and state.session.closed
        assert not any(call.args[0].startswith("mkdir") for call in box.exec.await_args_list)
        box.upload.assert_not_awaited()
    finally:
        release.set()
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        await state.session.close(timeout=0.5)


async def test_shorter_tool_timeout_remains_an_observation_after_confirmed_cleanup(monkeypatch):
    from nemo_gym.agents import SandboxSession
    from nemo_gym.agents.utils import sandbox_session
    from nemo_gym.sandbox import SandboxExecResult

    runner, state, box = _tool_loop(monkeypatch, budget=10)
    box.exec.side_effect = [
        SandboxExecResult(stdout="", stderr="", return_code=0),
        SandboxExecResult(stdout="", stderr="", error_type="timeout", return_code=124),
    ]
    monkeypatch.setattr(
        sandbox_session,
        "stop_and_confirm_cleanup",
        AsyncMock(return_value={"cleanup_confirmed": True, "return_code": 124, "timed_out": True, "error": None}),
    )
    monkeypatch.setattr(SandboxSession, "read_output_log", AsyncMock(return_value="partial output"))
    text = await runner._execute_tool(
        state, "bash", '{"command":"sleep 2"}', deadline=asyncio.get_running_loop().time() + 10
    )
    assert "Error: tool exited with 124" in text and "partial output" in text
    assert state.session.cleanup_confirmed and not state.session.closing
    box.exec.side_effect = None
    await state.session.close(timeout=0.5)


@pytest.mark.parametrize("failure", ["provider-result", "provider-exception", "unconfirmed-timeout"])
async def test_provider_failures_propagate_even_after_confirmed_cleanup(monkeypatch, failure):
    from nemo_gym.agents import SandboxSession
    from nemo_gym.agents.utils import sandbox_session
    from nemo_gym.sandbox import SandboxExecResult

    runner, state, box = _tool_loop(monkeypatch, budget=10)
    runner.config.max_turns = 1
    failures = {
        "provider-result": (
            SandboxExecResult(
                stdout="", stderr="transport unavailable", error_type="connection_error", return_code=125
            ),
            RuntimeError,
        ),
        "provider-exception": (ConnectionError("transport unavailable"), ConnectionError),
        "unconfirmed-timeout": (TimeoutError("transport timed out"), TimeoutError),
    }
    failed_exec, expected_error = failures[failure]
    box.exec.side_effect = [SandboxExecResult(stdout="", stderr="", return_code=0), failed_exec]
    monkeypatch.setattr(
        sandbox_session,
        "stop_and_confirm_cleanup",
        AsyncMock(return_value={"cleanup_confirmed": True, "return_code": 0, "timed_out": False, "error": None}),
    )
    monkeypatch.setattr(SandboxSession, "read_output_log", AsyncMock(return_value="partial output"))
    try:
        with pytest.raises(expected_error):
            await runner._run_loop(state, NeMoGymResponseCreateParamsNonStreaming(input=[]), "/v1/responses", {})
        assert state.session.cleanup_confirmed
        assert state.session.artifacts == "partial output"
        runner.server_client.post.assert_awaited_once()
        box.stop.assert_not_awaited()
    finally:
        box.exec.side_effect = None
        await state.session.close(timeout=0.5)


async def test_connection_setup_consumes_the_agent_budget(monkeypatch):
    from responses_api_agents.legal_agent_bench_native import app

    runner = agent()
    runner.config.agent_timeout_seconds = 0.01
    runner.server_client.post = AsyncMock()
    provider = SimpleNamespace(aclose=AsyncMock())

    async def attach(*args, **kwargs):
        await asyncio.sleep(0.03)
        return SimpleNamespace()

    monkeypatch.setattr(app, "get_global_config_dict", lambda: {"sandbox": {"docker": {}}})
    monkeypatch.setattr(app, "create_provider", lambda _: provider)
    monkeypatch.setattr(app.AsyncSandbox, "connect", attach)
    state = await runner._seed_agent_session_state(
        seed(
            sandbox_access={
                "connection": {"provider_config_ref": "sandbox", "descriptor": {"sandbox_id": "task"}},
                "workdir": "/workspace",
            }
        )
    )
    result = await runner._run_loop(state, NeMoGymResponseCreateParamsNonStreaming(input=[]), "/v1/responses", {})
    assert result.status == "incomplete"
    assert result.metadata["lab_finish_reason"] == "agent_timeout"
    runner.server_client.post.assert_not_awaited()
