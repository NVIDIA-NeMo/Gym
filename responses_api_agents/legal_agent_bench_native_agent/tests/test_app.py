# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Gym-native Legal Agent Bench loop."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.legal_agent_bench_native_agent import app


_REAL_PREFLIGHT = app.LabToolExecutor.preflight


def _config(**overrides) -> app.LegalAgentBenchNativeAgentConfig:
    values = {
        "host": "0.0.0.0",
        "port": 10000,
        "name": "legal_agent_bench_native_agent",
        "entrypoint": "app.py",
        "resources_server": ResourcesServerRef(name="lab", type="resources_servers"),
        "model_server": ModelServerRef(name="policy_model", type="responses_api_models"),
    }
    values.update(overrides)
    return app.LegalAgentBenchNativeAgentConfig(**values)


def _model_response(output: list[dict], *, response_id: str = "response", usage: bool = True) -> dict:
    payload = {
        "id": response_id,
        "created_at": 1,
        "model": "policy",
        "object": "response",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }
    if usage:
        payload["usage"] = {
            "input_tokens": 3,
            "input_tokens_details": {"cached_tokens": 1},
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 1},
            "total_tokens": 5,
        }
    return payload


def _function_call(name: str = "glob", arguments: str = '{"pattern":"**/*"}') -> dict:
    return {
        "id": "fc-1",
        "call_id": "call-1",
        "name": name,
        "arguments": arguments,
        "type": "function_call",
        "status": "completed",
    }


def _assistant_message(text: str = "Done") -> dict:
    return {
        "id": "message-1",
        "content": [{"annotations": [], "text": text, "type": "output_text"}],
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }


def _raw_response(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(ok=True, read=AsyncMock(return_value=json.dumps(payload).encode()), cookies={})


def _process(*, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        pid=123,
        returncode=returncode,
        communicate=AsyncMock(return_value=(stdout, stderr)),
    )


def _agent(*, max_turns: int = 60) -> app.LegalAgentBenchNativeAgent:
    client = MagicMock(spec=ServerClient)
    return app.LegalAgentBenchNativeAgent(config=_config(max_turns=max_turns), server_client=client)


@pytest.fixture(autouse=True)
def _successful_preflight(monkeypatch) -> None:
    monkeypatch.setattr(app.LabToolExecutor, "preflight", AsyncMock())


async def test_tool_loop_returns_full_responses_trajectory_and_usage(monkeypatch) -> None:
    agent = _agent()
    agent.server_client.post = AsyncMock(
        side_effect=[
            _raw_response(_model_response([_function_call()], response_id="tool-turn")),
            _raw_response(_model_response([_assistant_message()], response_id="final-turn")),
        ]
    )
    execute = AsyncMock(return_value="contract.docx")
    monkeypatch.setattr(app.LabToolExecutor, "execute", execute)
    body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Do the task"}])

    result = await agent.responses(SimpleNamespace(path_params={}), body)

    assert result.status == "completed"
    assert result.error is None
    assert result.usage.total_tokens == 10
    assert [type(item) for item in result.output] == [
        NeMoGymResponseFunctionToolCall,
        NeMoGymFunctionCallOutput,
        NeMoGymResponseOutputMessage,
    ]
    execute.assert_awaited_once_with("glob", '{"pattern":"**/*"}')
    second_input = agent.server_client.post.await_args_list[1].kwargs["json"].input
    assert isinstance(second_input[-1], NeMoGymFunctionCallOutput)
    assert second_input[-1].output == "contract.docx"


async def test_model_failure_preserves_partial_trajectory(monkeypatch) -> None:
    agent = _agent()
    agent.server_client.post = AsyncMock(
        side_effect=[
            _raw_response(_model_response([_function_call()])),
            RuntimeError("model disconnected"),
        ]
    )
    monkeypatch.setattr(app.LabToolExecutor, "execute", AsyncMock(return_value="file.txt"))

    result = await agent.responses(
        SimpleNamespace(path_params={}),
        NeMoGymResponseCreateParamsNonStreaming(input="Do the task"),
    )

    assert result.status == "failed"
    assert result.error is not None
    assert "model disconnected" in result.error.message
    assert any(isinstance(item, NeMoGymResponseFunctionToolCall) for item in result.output)
    assert any(isinstance(item, NeMoGymFunctionCallOutput) for item in result.output)


async def test_model_timeout_sets_structured_failure_metadata(monkeypatch) -> None:
    agent = _agent()
    agent.server_client.post = AsyncMock(
        side_effect=[
            _raw_response(_model_response([_function_call()])),
            TimeoutError(),
        ]
    )
    monkeypatch.setattr(app.LabToolExecutor, "execute", AsyncMock(return_value="file.txt"))

    result = await agent.responses(
        SimpleNamespace(path_params={}),
        NeMoGymResponseCreateParamsNonStreaming(input="Do the task"),
    )

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.message == "LAB model call timed out after 1800s"
    assert result.metadata == {app.AGENT_FAILURE_CLASS_METADATA_KEY: "agent_timed_out"}
    assert any(isinstance(item, NeMoGymResponseFunctionToolCall) for item in result.output)
    assert any(isinstance(item, NeMoGymFunctionCallOutput) for item in result.output)


@pytest.mark.parametrize(
    ("status", "expected_failure_class"),
    [
        (429, app.MODEL_CONNECTION_FAILURE_CLASS),
        (500, app.MODEL_CONNECTION_FAILURE_CLASS),
        (400, None),
    ],
)
async def test_model_http_failures_classify_only_retryable_statuses(status, expected_failure_class) -> None:
    agent = _agent()
    agent.server_client.post = AsyncMock(
        side_effect=app.aiohttp.ClientResponseError(
            request_info=MagicMock(real_url="http://policy/v1/responses"),
            history=(),
            status=status,
            message="model request failed",
        )
    )

    result = await agent.responses(
        SimpleNamespace(path_params={}),
        NeMoGymResponseCreateParamsNonStreaming(input="Do the task"),
    )

    assert result.status == "failed"
    assert result.error is not None
    assert f"ClientResponseError: {status}" in result.error.message
    assert (result.metadata or {}).get(app.AGENT_FAILURE_CLASS_METADATA_KEY) == expected_failure_class


async def test_turn_limit_is_a_scoreable_incomplete_outcome(monkeypatch) -> None:
    agent = _agent(max_turns=1)
    agent.server_client.post = AsyncMock(return_value=_raw_response(_model_response([_function_call()])))
    monkeypatch.setattr(app.LabToolExecutor, "execute", AsyncMock(return_value="file.txt"))

    result = await agent.responses(
        SimpleNamespace(path_params={}),
        NeMoGymResponseCreateParamsNonStreaming(input="Do the task"),
    )

    assert result.status == "incomplete"
    assert result.error is None
    assert result.incomplete_details.reason == "max_output_tokens"
    assert result.metadata == {"nemo_gym_stop_reason": "max_turns"}
    assert len(result.output) == 2


async def test_malformed_tool_arguments_are_returned_to_the_model(monkeypatch) -> None:
    executor = app.LabToolExecutor(timeout_seconds=1, max_output_chars=100)
    run = AsyncMock()
    monkeypatch.setattr(executor, "_run", run)

    result = await executor.execute("read", "not-json")

    assert result.startswith("Error: invalid JSON arguments")
    run.assert_not_awaited()


async def test_tool_executor_validates_calls_and_truncates_output(monkeypatch) -> None:
    executor = app.LabToolExecutor(timeout_seconds=1, max_output_chars=5)
    run = AsyncMock(return_value="complete output")
    monkeypatch.setattr(executor, "_run", run)

    assert await executor.execute("unknown", {}) == "Error: unknown tool: unknown"
    assert await executor.execute("read", []) == "Error: arguments for read must be a JSON object"
    assert await executor.execute("bash", {}) == "Error: command is required"
    assert await executor.execute("bash", {"command": "pwd"}) == "compl\n[output truncated]"
    run.assert_awaited_once_with(["/bin/bash", "-l", "-s"], stdin=b"pwd")

    run.reset_mock(return_value=True)
    run.return_value = "short"
    assert await executor.execute("read", {"file_path": "memo.txt"}) == "short"


async def test_preflight_rejects_container_tool_error(monkeypatch) -> None:
    executor = app.LabToolExecutor(timeout_seconds=1, max_output_chars=100)
    monkeypatch.setattr(executor, "execute", AsyncMock(return_value="Error: missing document tooling"))

    with pytest.raises(RuntimeError, match="missing document tooling"):
        await _REAL_PREFLIGHT(executor)


async def test_preflight_uses_its_dedicated_timeout(monkeypatch) -> None:
    executor = app.LabToolExecutor(
        timeout_seconds=60,
        preflight_timeout_seconds=120,
        max_output_chars=100,
    )
    run = AsyncMock(return_value="OK")
    monkeypatch.setattr(executor, "_run", run)

    await _REAL_PREFLIGHT(executor)

    run.assert_awaited_once_with(
        ["/usr/local/bin/python", str(app.CONTAINER_TOOL_RUNNER), "preflight"],
        stdin=b"{}",
        timeout_seconds=120,
    )


async def test_explicit_full_read_is_not_truncated(monkeypatch) -> None:
    executor = app.LabToolExecutor(timeout_seconds=1, max_output_chars=5)
    monkeypatch.setattr(executor, "_run", AsyncMock(return_value="complete document"))

    result = await executor.execute("read", {"file_path": "memo.txt", "limit": 0})

    assert result == "complete document"


async def test_tool_executor_passes_large_non_bash_payload_over_stdin(monkeypatch) -> None:
    executor = app.LabToolExecutor(timeout_seconds=1, max_output_chars=2_000_000)
    run = AsyncMock(return_value="written")
    monkeypatch.setattr(executor, "_run", run)
    content = "x" * 1_000_000

    result = await executor.execute("write", {"file_path": "memo.txt", "content": content})

    assert result == "written"
    command = run.await_args.args[0]
    assert command == ["/usr/local/bin/python", str(app.CONTAINER_TOOL_RUNNER), "write"]
    assert json.loads(run.await_args.kwargs["stdin"]) == {"file_path": "memo.txt", "content": content}


async def test_tool_executor_passes_large_bash_command_over_stdin(monkeypatch) -> None:
    executor = app.LabToolExecutor(timeout_seconds=1, max_output_chars=2_000_000)
    run = AsyncMock(return_value="written")
    monkeypatch.setattr(executor, "_run", run)
    command = "printf %s " + "x" * 1_000_000

    result = await executor.execute("bash", {"command": command})

    assert result == "written"
    assert run.await_args.args[0] == ["/bin/bash", "-l", "-s"]
    assert run.await_args.kwargs["stdin"] == command.encode()


async def test_process_runner_parses_container_result_and_stderr(monkeypatch) -> None:
    executor = app.LabToolExecutor(timeout_seconds=1, max_output_chars=100)
    process = _process(stdout=b'log line\n{"result":"document text"}\n', stderr=b"tool warning")
    create = AsyncMock(return_value=process)
    monkeypatch.setattr(app.asyncio, "create_subprocess_exec", create)

    result = await executor._run(
        ["/usr/local/bin/python", str(app.CONTAINER_TOOL_RUNNER), "read"],
        stdin=b'{"file_path":"memo.docx"}',
    )

    assert result == "document text\nSTDERR:\ntool warning"
    assert create.await_args.kwargs["cwd"] == "/workspace/output"
    assert create.await_args.kwargs["start_new_session"] is True
    assert create.await_args.kwargs["stdin"] is app.asyncio.subprocess.PIPE
    process.communicate.assert_awaited_once_with(input=b'{"file_path":"memo.docx"}')


@pytest.mark.parametrize(
    ("stdout", "returncode", "expected"),
    [
        (b"not json", 0, "not json"),
        (b"", 0, "(no output)"),
        (b"command failed", 2, "Error: command failed"),
    ],
)
async def test_process_runner_handles_unstructured_and_failed_output(
    monkeypatch,
    stdout: bytes,
    returncode: int,
    expected: str,
) -> None:
    executor = app.LabToolExecutor(timeout_seconds=1, max_output_chars=100)
    monkeypatch.setattr(
        app.asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=_process(stdout=stdout, returncode=returncode)),
    )

    result = await executor._run(
        ["/usr/local/bin/python", str(app.CONTAINER_TOOL_RUNNER), "preflight"],
        stdin=b"{}",
    )

    assert result == expected


async def test_process_runner_kills_timed_out_process_group(monkeypatch) -> None:
    executor = app.LabToolExecutor(timeout_seconds=7, max_output_chars=100)
    process = _process()
    monkeypatch.setattr(app.asyncio, "create_subprocess_exec", AsyncMock(return_value=process))

    async def time_out(awaitable, *, timeout):
        awaitable.close()
        assert timeout == 7
        raise asyncio.TimeoutError

    killpg = MagicMock(side_effect=ProcessLookupError)
    monkeypatch.setattr(app.asyncio, "wait_for", time_out)
    monkeypatch.setattr(app.os, "killpg", killpg)

    result = await executor._run(["/bin/bash", "-lc", "sleep 60"])

    assert result == "Error: tool timed out after 7s"
    killpg.assert_called_once_with(123, app.signal.SIGKILL)
    assert process.communicate.call_count == 2
    assert process.communicate.await_count == 1
    process.communicate.assert_any_call(input=None)


def test_merge_usage_ignores_missing_current_usage() -> None:
    response = app.NeMoGymResponse.model_validate(_model_response([]))

    assert app._merge_usage(response.usage, None) is response.usage


async def test_model_error_response_is_returned_with_trajectory() -> None:
    agent = _agent()
    payload = _model_response([_assistant_message("partial")])
    payload.update(
        {
            "status": "failed",
            "error": {"code": "server_error", "message": "policy backend failed"},
        }
    )
    agent.server_client.post = AsyncMock(return_value=_raw_response(payload))

    result = await agent.responses(
        SimpleNamespace(path_params={}),
        NeMoGymResponseCreateParamsNonStreaming(input="Do the task"),
    )

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.message == "policy backend failed"
    assert result.metadata == {
        app.AGENT_FAILURE_CLASS_METADATA_KEY: app.MODEL_CONNECTION_FAILURE_CLASS,
    }
    assert any(isinstance(item, NeMoGymResponseOutputMessage) for item in result.output)


@pytest.mark.parametrize("empty_output", [[], [_assistant_message("")]])
async def test_repeated_empty_model_responses_fail_after_two_nudges(empty_output) -> None:
    agent = _agent(max_turns=3)
    agent.server_client.post = AsyncMock(
        side_effect=[_raw_response(_model_response(empty_output, response_id=f"empty-{index}")) for index in range(3)]
    )

    result = await agent.responses(
        SimpleNamespace(path_params={}),
        NeMoGymResponseCreateParamsNonStreaming(input="Do the task"),
    )

    assert result.status == "failed"
    assert result.error is not None
    assert "repeatedly returned empty responses" in result.error.message
    second_input = agent.server_client.post.await_args_list[1].kwargs["json"].input
    assert any(getattr(item, "content", None) == app.INITIAL_EMPTY_RESPONSE_NUDGE for item in second_input)


async def test_preflight_failure_returns_failed_response(monkeypatch) -> None:
    agent = _agent()
    monkeypatch.setattr(app.LabToolExecutor, "preflight", AsyncMock(side_effect=RuntimeError("missing pandoc")))

    result = await agent.responses(
        SimpleNamespace(path_params={}),
        NeMoGymResponseCreateParamsNonStreaming(input="Do the task"),
    )

    assert result.status == "failed"
    assert result.error is not None
    assert "missing pandoc" in result.error.message
    agent.server_client.post.assert_not_called()
