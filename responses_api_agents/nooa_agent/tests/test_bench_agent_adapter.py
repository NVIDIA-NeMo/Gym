# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import asyncio
import json
import os
import re
import shlex
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nooa.tools.shell_tools import ShellTools

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseFunctionToolCall
from responses_api_agents.nooa_agent.bench_agent_adapter import invoke_bench_agent


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [None, RuntimeError("model failed"), asyncio.CancelledError()])
async def test_adapter_restores_task_cwd_and_always_drains_cleanup(tmp_path, monkeypatch, error):
    monkeypatch.chdir(tmp_path)
    value = object()
    agent = SimpleNamespace(
        context=SimpleNamespace(set=MagicMock()),
        shell=SimpleNamespace(cwd="/wrong", close=AsyncMock(), session=ShellTools(cwd=str(tmp_path)).session),
        _install_python_tools=MagicMock(),
        _solve_task=AsyncMock(return_value=value, side_effect=error),
        aclose=AsyncMock(),
    )
    request = NeMoGymResponseCreateParamsNonStreaming(input="unchanged canonical prompt")
    if error is None:
        assert await invoke_bench_agent(agent, request) is value
    else:
        with pytest.raises(type(error)):
            await invoke_bench_agent(agent, request)
    agent.shell.close.assert_awaited_once()
    agent._install_python_tools.assert_called_once_with(str(tmp_path))
    agent._solve_task.assert_awaited_once_with("unchanged canonical prompt")
    agent.context.set.assert_not_called()
    agent.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [None, RuntimeError("invocation failed"), asyncio.CancelledError()])
async def test_cleanup_failure_preserves_the_original_invocation_exception(tmp_path, monkeypatch, caplog, terminal):
    monkeypatch.chdir(tmp_path)
    cleanup_error = RuntimeError("cleanup also failed")
    agent = SimpleNamespace(
        context=SimpleNamespace(set=MagicMock()),
        shell=ShellTools(cwd=str(tmp_path)),
        _solve_task=AsyncMock(return_value="completed", side_effect=terminal),
        aclose=AsyncMock(side_effect=cleanup_error),
    )
    expected = terminal if terminal is not None else cleanup_error

    with pytest.raises(type(expected)) as caught:
        await invoke_bench_agent(agent, NeMoGymResponseCreateParamsNonStreaming(input="task"))

    assert caught.value is expected
    agent.aclose.assert_awaited_once()
    if terminal is not None:
        assert "BenchAgent cleanup failed" in caplog.text
        assert "cleanup also failed" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_kind", ["success", "failure", "cancelled"])
@pytest.mark.parametrize("same_error", [False, True])
async def test_sequential_cleanup_preserves_all_errors_and_original_exception(
    tmp_path, monkeypatch, terminal_kind: str, same_error: bool
) -> None:
    from responses_api_agents.nooa_agent import bench_agent_adapter

    monkeypatch.chdir(tmp_path)
    terminal = {
        "success": None,
        "failure": RuntimeError("original task failure"),
        "cancelled": asyncio.CancelledError(),
    }[terminal_kind]
    cause = ValueError("original cause")
    context = ValueError("original context")
    if terminal is not None:
        terminal.__cause__ = cause
        terminal.__context__ = context
    first = ConnectionError("agent close failed")
    second = first if same_error else ValueError("scope close failed")

    @asynccontextmanager
    async def failing_scope():
        yield SimpleNamespace(adopt=lambda session: None)
        raise second

    monkeypatch.setattr(bench_agent_adapter, "preserve_background_services", failing_scope)
    agent = SimpleNamespace(
        context=SimpleNamespace(set=MagicMock()),
        shell=SimpleNamespace(cwd=str(tmp_path), session=object()),
        _solve_task=AsyncMock(return_value="done", side_effect=terminal),
        aclose=AsyncMock(side_effect=first),
    )
    expected = first if terminal is None else terminal
    with pytest.raises(type(expected)) as caught:
        await invoke_bench_agent(agent, NeMoGymResponseCreateParamsNonStreaming(input="task"))

    assert caught.value is expected
    assert caught.value.nooa_cleanup_errors == (first, second)
    if terminal is not None:
        assert terminal.__cause__ is cause
        assert terminal.__context__ is context
    agent.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["entry", "adoption", "cwd"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_setup_failures_still_close_agent_and_remain_setup_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, cleanup_fails: bool
) -> None:
    from responses_api_agents.nooa_agent import bench_agent_adapter
    from responses_api_agents.nooa_agent.termination import cleanup_errors

    setup_error = AttributeError("invalid shell setup")
    cleanup_error = ConnectionError("close after setup failed")
    scope_closed = []

    def adopt(session):
        if stage == "adoption":
            raise setup_error

    @asynccontextmanager
    async def scope():
        if stage == "entry":
            raise setup_error
        try:
            yield SimpleNamespace(adopt=adopt)
        finally:
            scope_closed.append(True)

    monkeypatch.setattr(bench_agent_adapter, "preserve_background_services", scope)
    agent = SimpleNamespace(
        shell=SimpleNamespace(cwd="/wrong", session=object(), close=AsyncMock(side_effect=setup_error)),
        _solve_task=AsyncMock(),
        aclose=AsyncMock(side_effect=cleanup_error if cleanup_fails else None),
    )
    with pytest.raises(AttributeError) as caught:
        await invoke_bench_agent(agent, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert caught.value is setup_error
    assert cleanup_errors(setup_error) == ((cleanup_error,) if cleanup_fails else ())
    agent.aclose.assert_awaited_once()
    agent._solve_task.assert_not_awaited()
    assert scope_closed == ([] if stage == "entry" else [True])


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["agent", "scope"])
@pytest.mark.parametrize("prior", ["success", "task", "setup"])
async def test_cleanup_cancellation_preserves_original_failure_and_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, boundary: str, prior: str
) -> None:
    from responses_api_agents.nooa_agent import bench_agent_adapter
    from responses_api_agents.nooa_agent.termination import cleanup_errors

    monkeypatch.chdir(tmp_path)
    original = ValueError("original task or setup failure") if prior != "success" else None
    original_cause = RuntimeError("underlying cause")
    if original is not None:
        original.__cause__ = original_cause
    scope_closed = []

    async def cancel_cleanup():
        deadline.reschedule(asyncio.get_running_loop().time())
        await asyncio.Event().wait()

    def adopt(session):
        if prior == "setup":
            raise original

    @asynccontextmanager
    async def scope():
        yield SimpleNamespace(adopt=adopt)
        scope_closed.append(True)
        if boundary == "scope":
            await cancel_cleanup()

    monkeypatch.setattr(bench_agent_adapter, "preserve_background_services", scope)
    agent = SimpleNamespace(
        shell=SimpleNamespace(cwd=str(tmp_path), session=object()),
        _solve_task=AsyncMock(return_value="done", side_effect=original if prior == "task" else None),
        aclose=AsyncMock(side_effect=cancel_cleanup if boundary == "agent" else None),
    )
    with pytest.raises(TimeoutError) as caught:
        async with asyncio.timeout(None) as deadline:
            await invoke_bench_agent(agent, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    cancelled = caught.value.__cause__
    assert isinstance(cancelled, asyncio.CancelledError)
    assert cancelled.__cause__ is original
    assert cleanup_errors(cancelled) == (cancelled,)
    if original is not None:
        assert original.__cause__ is original_cause
        assert str(original) in caplog.text
        assert str(original_cause) in caplog.text
    agent.aclose.assert_awaited_once()
    assert scope_closed == [True]


@pytest.mark.asyncio
async def test_delegated_fresh_shell_service_is_usable_during_parent_task(tmp_path, monkeypatch):
    pytest.importorskip("nooa_bench.bench_agent")
    from responses_api_agents.nooa_agent.invocation import NOOAInvocationConfig
    from responses_api_agents.nooa_agent.runner import InProcessNOOARunner, NOOARunRequest
    from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response

    monkeypatch.chdir(tmp_path)
    (tmp_path / "marker.txt").write_text("ready during parent task")
    (tmp_path / "service.py").write_text(
        "import signal\n"
        "from http.server import HTTPServer, SimpleHTTPRequestHandler\n"
        "from pathlib import Path\n"
        "signal.alarm(30)\n"
        "server = HTTPServer(('127.0.0.1', 0), SimpleHTTPRequestHandler)\n"
        "Path('service.port').write_text(str(server.server_port))\n"
        "server.serve_forever()\n"
    )
    command = f"{shlex.quote(sys.executable)} service.py > service.log 2>&1 & echo $! > service.pid"
    code = iter(
        [
            "import asyncio\n"
            "from pathlib import Path\n"
            "child = await self.delegate('start the fixture service')\n"
            "assert child.evidence == 'service started'\n"
            "async with asyncio.timeout(5):\n"
            "    while not Path('service.port').exists():\n"
            "        await asyncio.sleep(0.01)\n"
            "    port = int(Path('service.port').read_text())\n"
            "    reader, writer = await asyncio.open_connection('127.0.0.1', port)\n"
            "    try:\n"
            "        writer.write(b'GET /marker.txt HTTP/1.0\\r\\nHost: localhost\\r\\n\\r\\n')\n"
            "        await writer.drain()\n"
            "        response = await reader.read()\n"
            "    finally:\n"
            "        writer.close()\n"
            "        await writer.wait_closed()\n"
            "assert b'200 OK' in response and response.endswith(b'ready during parent task')\n"
            "return_result(TaskResult(solution_description='parent', "
            "evidence='service used during parent task', how_to_verify='HTTP'))",
            "from nooa.tools.shell_tools import ShellTools\n"
            "fresh_shell = ShellTools(cwd=str(self.shell.cwd))\n"
            f"await fresh_shell.run({command!r})\n"
            "await fresh_shell.close()\n"
            "return_result(TaskResult(solution_description='child', evidence='service started', how_to_verify='HTTP'))",
        ]
    )
    calls = []

    async def post(**kwargs):
        calls.append(kwargs)
        output = NeMoGymResponseFunctionToolCall(
            id=f"f{len(calls)}",
            call_id=f"c{len(calls)}",
            name="python_cell",
            arguments=json.dumps({"code": next(code)}),
        )
        return FakeHTTPResponse(model_response(output, response_id=f"r{len(calls)}"))

    runner = InProcessNOOARunner(
        invocation=NOOAInvocationConfig(
            agent_class="nooa_bench.bench_agent:BenchAgent",
            invocation_adapter="responses_api_agents.nooa_agent.bench_agent_adapter:invoke_bench_agent",
        ),
        server_client=SimpleNamespace(post=post),
        model_server_name="policy",
        max_policy_calls=2,
    )
    try:
        result = await runner.run(
            NOOARunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="fixture"),
                model_url_path="/v1/responses",
            )
        )
        assert result.return_value.evidence == "service used during parent task"
        assert len(calls) == len(result.trajectory.turns) == 2
    finally:
        pid_path = tmp_path / "service.pid"
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [None, 2])
async def test_real_benchagent_delegation_uses_gym_trace_and_shared_call_budget(
    tmp_path, monkeypatch, limit: int | None
):
    pytest.importorskip("nooa_bench.bench_agent")
    from responses_api_agents.nooa_agent.invocation import NOOAInvocationConfig
    from responses_api_agents.nooa_agent.runner import InProcessNOOARunner, NOOARunRequest
    from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response

    monkeypatch.chdir(tmp_path)
    code = [
        "child = await self.delegate('bounded fixture')\nreturn_result(TaskResult(solution_description='parent', evidence=child.evidence, how_to_verify='fixture'))",
        "return_result(TaskResult(solution_description='child', evidence='delegated evidence', how_to_verify='fixture'))",
    ]
    calls = []

    async def post(**kwargs):
        calls.append(kwargs)
        output = NeMoGymResponseFunctionToolCall(
            id=f"f{len(calls)}",
            call_id=f"c{len(calls)}",
            name="python_cell",
            arguments=json.dumps({"code": code[len(calls) - 1]}),
        )
        return FakeHTTPResponse(model_response(output, response_id=f"r{len(calls)}"))

    client = SimpleNamespace(post=post)
    invocation = NOOAInvocationConfig(
        agent_class="nooa_bench.bench_agent:BenchAgent",
        invocation_adapter="responses_api_agents.nooa_agent.bench_agent_adapter:invoke_bench_agent",
    )
    runner = InProcessNOOARunner(
        invocation=invocation,
        server_client=client,
        model_server_name="policy",
        max_policy_calls=limit,
        context_window=262144,
    )
    result = await runner.run(
        NOOARunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="fixture", max_output_tokens=32768),
            model_url_path="/v1/responses",
        )
    )
    assert result.return_value.evidence == "delegated evidence"
    assert len(calls) == len(result.trajectory.turns) == 2
    assert len(result.trajectory.invocations) >= 2
    assert all(call["json"].max_output_tokens == 32768 for call in calls)
    assert result.termination_reason is None


@pytest.mark.asyncio
async def test_default_summarizer_calls_share_gym_budget_and_capture(tmp_path):
    module = pytest.importorskip("nooa_bench.bench_agent")
    from responses_api_agents.nooa_agent.gym_llm import GymResponsesLLM, PolicyCallBudgetExceeded, RolloutLLMState
    from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response

    state = RolloutLLMState(max_policy_calls=1)
    captured = []

    async def post(**kwargs):
        output = {
            "type": "message",
            "id": "summary",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": '{"value":"summary"}', "annotations": []}],
        }
        return FakeHTTPResponse(model_response(output, response_id="summary"))

    llm = GymResponsesLLM(
        server_client=SimpleNamespace(post=post),
        model_server_name="policy",
        model_url_path="/v1/responses",
        state=state,
        cookies={},
        on_call=captured.append,
        context_window=262144,
    )
    agent = module.BenchAgent(llm=llm, working_dir=str(tmp_path))
    try:
        assert agent._max_delegation_depth == 4 and len(agent._summarizers) == 1
        summarizer = agent._summarizers[0]
        assert summarizer.llm is llm
        assert await summarizer.summarize("synthetic history", 100) == "summary"
        assert len(captured) == len(state.calls) == 1
        with pytest.raises(PolicyCallBudgetExceeded):
            await llm.acall([{"role": "user", "content": "second"}])
        assert len(captured) == 1
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [None, RuntimeError("invocation failed"), asyncio.CancelledError()])
async def test_scope_cleanup_preserves_original_exception(tmp_path, monkeypatch, caplog, terminal):
    from nooa.tools._bash_session import BashSession

    monkeypatch.chdir(tmp_path)
    cleanup_error = RuntimeError("fresh shell cleanup failed")
    fresh_sessions = []
    original_close = BashSession.close

    async def solve(task):
        fresh_sessions.append(ShellTools(cwd=str(tmp_path)).session)
        if terminal is not None:
            raise terminal
        return "completed"

    async def close(session):
        await original_close(session)
        if session in fresh_sessions:
            raise cleanup_error

    monkeypatch.setattr(BashSession, "close", close)
    shell = ShellTools(cwd=str(tmp_path))
    agent = SimpleNamespace(
        shell=shell, context=SimpleNamespace(set=MagicMock()), _solve_task=solve, aclose=shell.close
    )
    expected = terminal if terminal is not None else cleanup_error
    with pytest.raises(type(expected)) as caught:
        await invoke_bench_agent(agent, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert caught.value is expected
    if terminal is not None:
        assert "fresh shell cleanup failed" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["agent", "scope"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("terminal", ["success", "policy_budget", "output_limit"])
async def test_real_cleanup_failure_remains_ungradable(tmp_path, monkeypatch, boundary, cleanup_fails, terminal):
    from nooa.tools._bash_session import BashSession

    module = pytest.importorskip("nooa_bench.bench_agent")
    from responses_api_agents.nooa_agent.invocation import NOOAInvocationConfig
    from responses_api_agents.nooa_agent.result import finalize_run_result
    from responses_api_agents.nooa_agent.runner import InProcessNOOARunner, NOOARunFailure, NOOARunRequest
    from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response

    monkeypatch.chdir(tmp_path)
    cleanup_error = RuntimeError("injected cleanup failure")
    fresh_sessions = []
    solve_errors = []
    cleanup_attempts = []
    original_solve = module.BenchAgent._solve_task
    original_close = BashSession.close
    original_aclose = ShellTools.close

    async def solve(agent, *args, **kwargs):
        fresh_sessions.append(ShellTools(cwd=str(tmp_path)).session)
        try:
            return await original_solve(agent, *args, **kwargs)
        except Exception as error:
            solve_errors.append(error)
            raise

    async def close(session):
        await original_close(session)
        if boundary == "scope" and session in fresh_sessions:
            cleanup_attempts.append(session)
            if cleanup_fails:
                raise cleanup_error

    async def aclose(agent):
        await original_aclose(agent)
        if boundary == "agent":
            cleanup_attempts.append(agent)
            if cleanup_fails:
                raise cleanup_error

    async def post(**kwargs):
        if terminal == "success":
            return FakeHTTPResponse(
                model_response(
                    NeMoGymResponseFunctionToolCall(
                        id="done",
                        call_id="done",
                        name="python_cell",
                        arguments=json.dumps(
                            {
                                "code": "return_result(TaskResult(solution_description='done', "
                                "evidence='completed', how_to_verify='fixture'))"
                            }
                        ),
                    )
                )
            )
        response = model_response(response_id="truncated")
        response.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        return FakeHTTPResponse(response)

    monkeypatch.setattr(module.BenchAgent, "_solve_task", solve)
    monkeypatch.setattr(ShellTools, "close", aclose)
    monkeypatch.setattr(BashSession, "close", close)
    runner = InProcessNOOARunner(
        invocation=NOOAInvocationConfig(
            agent_class="nooa_bench.bench_agent:BenchAgent",
            invocation_adapter="responses_api_agents.nooa_agent.bench_agent_adapter:invoke_bench_agent",
        ),
        server_client=SimpleNamespace(post=post),
        model_server_name="policy",
        max_policy_calls=0 if terminal == "policy_budget" else None,
    )
    request = NOOARunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
        model_url_path="/v1/responses",
    )
    if cleanup_fails:
        with pytest.raises(NOOARunFailure) as caught:
            await runner.run(request)
        assert caught.value.__cause__ is (cleanup_error if terminal == "success" else solve_errors[0])
        if terminal == "success":
            assert str(caught.value) == "Cleanup failed: RuntimeError: injected cleanup failure"
        result = caught.value.result
        assert result.termination_reason is None
    else:
        result = await runner.run(request)
        expected = {
            "success": None,
            "policy_budget": "policy_budget_exceeded",
            "output_limit": "max_output_tokens_exceeded",
        }[terminal]
        assert result.termination_reason == expected
        response, _ = finalize_run_result(result)
        assert response.status == ("completed" if terminal == "success" else "incomplete")
        assert response.error is None
    if terminal == "policy_budget":
        solve_invocations = [
            invocation
            for invocation in result.trajectory.invocations
            if invocation.error_type == type(solve_errors[0]).__name__
        ]
        assert len(solve_invocations) == 1
        assert solve_invocations[0].status == ("failed" if cleanup_fails else "incomplete")
    assert result.trajectory.invocations[0].status == (
        "failed" if cleanup_fails else "completed" if terminal == "success" else "incomplete"
    )
    assert len(cleanup_attempts) == 1
    assert len(solve_errors) == (0 if terminal == "success" else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["agent", "scope"])
async def test_timeout_during_cleanup_overrides_policy_termination(tmp_path, monkeypatch, boundary):
    from nooa.tools._bash_session import BashSession

    module = pytest.importorskip("nooa_bench.bench_agent")
    from responses_api_agents.nooa_agent.invocation import NOOAInvocationConfig
    from responses_api_agents.nooa_agent.runner import InProcessNOOARunner, NOOARunRequest

    monkeypatch.chdir(tmp_path)
    fresh_sessions = []
    cleanup_started = []
    original_solve = module.BenchAgent._solve_task
    original_close = BashSession.close
    original_aclose = ShellTools.close

    async def interrupt_cleanup():
        cleanup_started.append(boundary)
        deadline.reschedule(asyncio.get_running_loop().time())
        await asyncio.Event().wait()

    async def solve(agent, *args, **kwargs):
        fresh_sessions.append(ShellTools(cwd=str(tmp_path)).session)
        return await original_solve(agent, *args, **kwargs)

    async def close(session):
        await original_close(session)
        if boundary == "scope" and session in fresh_sessions:
            await interrupt_cleanup()

    async def aclose(agent):
        await original_aclose(agent)
        if boundary == "agent":
            await interrupt_cleanup()

    monkeypatch.setattr(module.BenchAgent, "_solve_task", solve)
    monkeypatch.setattr(ShellTools, "close", aclose)
    monkeypatch.setattr(BashSession, "close", close)
    runner = InProcessNOOARunner(
        invocation=NOOAInvocationConfig(
            agent_class="nooa_bench.bench_agent:BenchAgent",
            invocation_adapter="responses_api_agents.nooa_agent.bench_agent_adapter:invoke_bench_agent",
        ),
        server_client=SimpleNamespace(),
        model_server_name="policy",
        max_policy_calls=0,
    )
    with pytest.raises(TimeoutError) as caught:
        async with asyncio.timeout(None) as deadline:
            await runner.run(
                NOOARunRequest(
                    responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
                    model_url_path="/v1/responses",
                )
            )
    assert cleanup_started == [boundary]
    assert isinstance(caught.value.__cause__, asyncio.CancelledError)
    assert caught.value.__cause__.nooa_result.termination_reason is None


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_fails", [False, True])
async def test_real_benchagent_loop_guard_preserves_gradable_evidence(tmp_path, monkeypatch, tool_fails):
    pytest.importorskip("nooa_bench.bench_agent")
    from responses_api_agents.nooa_agent.invocation import NOOAInvocationConfig
    from responses_api_agents.nooa_agent.result import finalize_run_result
    from responses_api_agents.nooa_agent.runner import InProcessNOOARunner, NOOARunRequest
    from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response

    monkeypatch.chdir(tmp_path)
    calls = []
    code = "raise ValueError('repeated failure')" if tool_fails else "print('repeated success')"

    async def post(**kwargs):
        calls.append(kwargs)
        output = NeMoGymResponseFunctionToolCall(
            id=f"f{len(calls)}",
            call_id=f"c{len(calls)}",
            name="python_cell",
            arguments=json.dumps({"code": code}),
        )
        return FakeHTTPResponse(model_response(output, response_id=f"r{len(calls)}"))

    runner = InProcessNOOARunner(
        invocation=NOOAInvocationConfig(
            agent_class="nooa_bench.bench_agent:BenchAgent",
            invocation_adapter="responses_api_agents.nooa_agent.bench_agent_adapter:invoke_bench_agent",
        ),
        server_client=SimpleNamespace(post=post),
        model_server_name="policy",
        max_policy_calls=8,
    )
    result = await runner.run(
        NOOARunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
            model_url_path="/v1/responses",
        )
    )
    assert result.termination_reason == "loop_detected"
    assert result.trajectory.invocations[0].status == "incomplete"
    assert len(calls) == len(result.trajectory.turns) == 8
    response, observations = finalize_run_result(result)
    assert response.status == "incomplete" and response.error is None
    expected_calls = {f"c{index}" for index in range(1, 9)}
    native_prefills = {
        item.call_id for item in calls[0]["json"].input if item.type == "function_call" and item.name == "python_cell"
    }
    assert native_prefills
    assert {
        item.call_id for item in response.output if item.type == "function_call" and item.name == "python_cell"
    } == expected_calls | native_prefills
    # Failed repeats are blocked before execution on calls 7 and 8; the guard's feedback is their output.
    actual_outputs = {item.call_id for item in response.output if item.type == "function_call_output"}
    assert actual_outputs & expected_calls == expected_calls
    tools = {tool.tool_call_id: tool for tool in result.trajectory.tool_calls}
    blocked = {call_id for call_id, tool in tools.items() if tool.error_type == "ToolCallRejected"}
    assert blocked == ({"c7", "c8"} if tool_fails else set())
    for call_id in blocked:
        assert tools[call_id].status == "failed"
        assert tools[call_id].tool_name == "python_cell"
        # The output is the guard's feedback as the model read it, not an execution result.
        assert isinstance(tools[call_id].output, str) and tools[call_id].output
        assert "repeated failure" not in tools[call_id].output
    assert "loop_detected" in {gap.code for gap in observations.gaps}


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_boundary", ["embedded", "worker"])
@pytest.mark.parametrize("cleanup_boundary", ["agent", "scope", "both"])
@pytest.mark.parametrize("error_type", [ConnectionError, TimeoutError, ValueError])
async def test_cleanup_failure_retry_and_diagnostic_cross_runtime_boundaries(
    tmp_path, monkeypatch, wire_boundary: str, cleanup_boundary: str, error_type: type[Exception]
) -> None:
    from fastapi import HTTPException
    from nooa.tools._bash_session import BashSession

    module = pytest.importorskip("nooa_bench.bench_agent")
    from nemo_gym.server_utils import ServerClient
    from responses_api_agents.nooa_agent import sandbox_entrypoint as entrypoint
    from responses_api_agents.nooa_agent.app import NOOAAgent
    from responses_api_agents.nooa_agent.invocation import NOOAInvocationConfig
    from responses_api_agents.nooa_agent.runner import NOOARunFailure
    from responses_api_agents.nooa_agent.termination import NOOATerminationReason
    from responses_api_agents.nooa_agent.tests.test_app import close, open_session
    from responses_api_agents.nooa_agent.tests.test_config import agent_config
    from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response
    from responses_api_agents.nooa_agent.tests.test_sandbox_entrypoint import payload

    monkeypatch.chdir(tmp_path)
    cleanup_error = error_type(
        "secondary cleanup connection failed" + ("x" * 2500 if cleanup_boundary == "both" else "")
    )
    scope_error = (ConnectionError if error_type is ValueError else ValueError)("scope close failed" + "y" * 2500)
    fresh_sessions = []
    solve_errors = []
    original_solve = module.BenchAgent._solve_task
    original_session_close = BashSession.close
    original_shell_close = ShellTools.close

    async def solve(agent, *args, **kwargs):
        fresh_sessions.append(ShellTools(cwd=str(tmp_path)).session)
        try:
            return await original_solve(agent, *args, **kwargs)
        except Exception as error:
            solve_errors.append(error)
            raise

    async def session_close(session):
        await original_session_close(session)
        if session in fresh_sessions:
            if cleanup_boundary == "scope":
                raise cleanup_error
            if cleanup_boundary == "both":
                raise scope_error

    async def shell_close(shell):
        await original_shell_close(shell)
        if cleanup_boundary in {"agent", "both"}:
            raise cleanup_error

    calls = []

    async def post(self=None, **kwargs):
        calls.append(kwargs)
        return FakeHTTPResponse(
            model_response(
                NeMoGymResponseFunctionToolCall(
                    id="progress", call_id="c1", name="python_cell", arguments='{"code":"print(1)"}'
                )
            )
        )

    monkeypatch.setattr(module.BenchAgent, "_solve_task", solve)
    monkeypatch.setattr(BashSession, "close", session_close)
    monkeypatch.setattr(ShellTools, "close", shell_close)
    invocation = NOOAInvocationConfig(
        agent_class="nooa_bench.bench_agent:BenchAgent",
        invocation_adapter="responses_api_agents.nooa_agent.bench_agent_adapter:invoke_bench_agent",
    )
    transient = cleanup_boundary == "both" or error_type is not ValueError
    if wire_boundary == "worker":
        monkeypatch.setattr(entrypoint._ModelClient, "post", post)
        request = payload()
        request.invocation = invocation
        request.max_policy_calls = 1
        artifact = await entrypoint.execute(request)
        restored = entrypoint.SandboxResult.model_validate_json(artifact.model_dump_json())
        assert restored.error.kind == ("transient" if transient else "fatal")
        assert restored.termination_reason is NOOATerminationReason.INFRASTRUCTURE_ERROR
        assert restored.response.status == "failed"
        assert restored.response.error is not None
        assert restored.response.output
        diagnostic = restored.error.message
        assert restored.termination_error == diagnostic
        assert restored.response.error.message == diagnostic
    else:
        client = MagicMock(spec=ServerClient)
        client.post = AsyncMock(side_effect=post)
        instance = NOOAAgent(
            config=agent_config(nooa=invocation.model_dump(), max_policy_calls=1), server_client=client
        )
        request = await open_session(instance)
        try:
            with pytest.raises(HTTPException if transient else NOOARunFailure) as caught:
                await instance.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
            failure = caught.value.__cause__ if transient else caught.value
            assert isinstance(failure, NOOARunFailure)
            assert failure.__cause__ is solve_errors[0]
            if transient:
                assert caught.value.status_code == 503
            state = instance._require_agent_session("session")
            assert state.result.termination_reason is NOOATerminationReason.INFRASTRUCTURE_ERROR
            diagnostic = state.result.termination_error
        finally:
            receipt = await close(instance)
        assert any(
            gap.code == NOOATerminationReason.INFRASTRUCTURE_ERROR and gap.detail == diagnostic
            for gap in receipt.agent_observations.gaps
        )

    assert len(calls) == len(solve_errors) == 1
    expected_errors = (cleanup_error, scope_error) if cleanup_boundary == "both" else (cleanup_error,)
    assert solve_errors[0].nooa_cleanup_errors == expected_errors
    assert not hasattr(solve_errors[0], "nooa_cleanup_error")
    assert str(solve_errors[0]) in diagnostic
    for error in expected_errors:
        detail = str(error)[:100] if cleanup_boundary == "both" else str(error)
        assert f"{type(error).__name__}: {detail}" in diagnostic
    if wire_boundary == "worker":
        assert len(diagnostic) <= 2000


@pytest.mark.asyncio
@pytest.mark.parametrize("context_limit", [None, 1])
async def test_real_benchagent_preserves_full_task_across_model_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, context_limit: int | None
) -> None:
    module = pytest.importorskip("nooa_bench.bench_agent")
    from nooa.config import TruncationConfig

    from responses_api_agents.nooa_agent.gym_llm import GymResponsesLLM, RolloutLLMState
    from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response

    monkeypatch.chdir(tmp_path)
    middle = "MIDDLE_REQUIREMENT: preserve use_netrc and finish tool instructions."
    task = (
        "  Original task\n"
        + "Both delimiters: \"\"\" and '''\n"
        + ("context α 'quoted' \\ path\n" * 1500)
        + middle
        + ("\nremaining context" * 2000)
        + "\n  "
    )
    volatile = "VOLATILE_CONTEXT_SHOULD_BE_EVICTED"
    calls = []
    code = [
        "print('continue to the next model call')",
        "return_result(TaskResult(solution_description='done', evidence='two calls', how_to_verify='fixture'))",
    ]

    async def post(**kwargs):
        calls.append(kwargs["json"])
        output = NeMoGymResponseFunctionToolCall(
            id=f"f{len(calls)}",
            call_id=f"c{len(calls)}",
            name="python_cell",
            arguments=json.dumps({"code": code[len(calls) - 1]}),
        )
        return FakeHTTPResponse(model_response(output, response_id=f"r{len(calls)}"))

    llm = GymResponsesLLM(
        server_client=SimpleNamespace(post=post),
        model_server_name="policy",
        model_url_path="/v1/responses",
        state=RolloutLLMState(max_policy_calls=2),
        cookies={},
        context_window=262144,
    )
    agent = module.BenchAgent(
        llm=llm,
        working_dir=str(tmp_path),
        truncation=TruncationConfig(max_context_tokens=context_limit),
    )
    agent.context.set("volatile_probe", volatile)
    result = await invoke_bench_agent(agent, NeMoGymResponseCreateParamsNonStreaming(input=task))

    assert result.evidence == "two calls"
    assert len(calls) == 2
    for call in calls:
        # These are the actual requests after native argument/context rendering.
        instructions = call.instructions or ""
        assert middle in instructions
        assert instructions.count(middle) == 1
        assert call.model_dump_json().count(middle) == 1
        task_blocks = re.findall(r"<task(?: [^>]*)?>\n(.*?)\n</task>", instructions, flags=re.DOTALL)
        assert len(task_blocks) == 1
        parameters = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(task_blocks[0]).body
            if isinstance(node, ast.Assign)
        }
        assert parameters["description"] == task
        assert "<benchmark_task>" not in instructions
        assert (volatile in call.model_dump_json()) == (context_limit is None)
    assert "benchmark_task" not in agent.context
