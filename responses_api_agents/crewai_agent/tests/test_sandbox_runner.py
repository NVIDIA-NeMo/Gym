# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sandbox runner's shell tool and transcript, with real bash processes in a temporary directory."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

from responses_api_agents.crewai_agent.sandbox_runner import Shell, Transcript


@pytest.mark.parametrize(
    "retry, retry_after, expected",
    [
        (0, "0.25", 0.25),
        (4, "120", 60.0),
        (0, "Thu, 01 Jan 2099 00:00:00 GMT", 60.0),
        (0, "Thu, 01 Jan 1970 00:00:00 GMT", 1.0),
        (2, None, 4.0),
        (2, "invalid", 4.0),
        (2, "nan", 4.0),
        (2, "inf", 4.0),
        (2, "-1", 4.0),
        (2, "0", 4.0),
        (10000, None, 60.0),
    ],
)
def test_retry_after_dates_and_seconds_are_bounded_and_invalid_values_use_backoff(
    monkeypatch, retry, retry_after, expected
):
    from responses_api_agents.crewai_agent import sandbox_runner

    monkeypatch.setattr(sandbox_runner.random, "uniform", lambda low, high: high)
    assert sandbox_runner._retry_wait_seconds(retry, retry_after) == expected


def test_infrastructure_cause_capture_keeps_status_and_types_without_request_secrets():
    import httpx

    from responses_api_agents.crewai_agent.sandbox_runner import InfrastructureError

    request = httpx.Request("GET", "https://example.com/mcp?token=private", headers={"Authorization": "private"})
    response = httpx.Response(401, request=request, text="private response body")
    cause = httpx.HTTPStatusError("private diagnostic with request URL", request=request, response=response)
    failure = InfrastructureError("required MCP grant weather did not list its tools", operation="mcp_list_tools")
    failure.__cause__ = ExceptionGroup("private SDK wrapper", [cause])

    report = failure.to_report()

    assert [cause["exception_type"] for cause in report["causes"]] == [
        "builtins.ExceptionGroup",
        "httpx.HTTPStatusError",
    ]
    assert report["causes"][-1]["http_status"] == 401
    assert "private" not in json.dumps(report) and "example.com" not in json.dumps(report)


# Killing a process group is reliable on Linux; macOS's bash 3.2 sometimes leaves a subshell of a killed group running.
_LINUX_PROCESS_GROUPS = pytest.mark.skipif(sys.platform != "linux", reason="Linux process-group kills")


@pytest.fixture
def shell(tmp_path) -> Shell:
    workdir = tmp_path / "task"
    workdir.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    # Room for a loaded machine; the deadline tests shorten it.
    return Shell(
        workdir=str(workdir),
        env={"PATH": "/usr/bin:/bin", "TASK_LABEL": "owner"},
        timeout_seconds=10,
        max_output_bytes=1000,
        scratch=scratch,
    )


async def _eventually(predicate, *, timeout: float = 10) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.05)


async def test_a_command_runs_in_a_fresh_bash_in_the_task_directory_with_its_output_merged(shell):
    text, status, error_type = await asyncio.wait_for(
        shell.run('pwd; echo "out $TASK_LABEL"; echo err >&2; exit 3'), 10
    )

    assert text == f"{shell.workdir}\nout owner\nerr\n\n[exit code: 3]"
    assert (status, error_type) == ("failed", "exit_code_3")


async def test_a_command_that_succeeds_reports_no_error(shell):
    assert await asyncio.wait_for(shell.run("echo done"), 10) == ("done\n\n[exit code: 0]", "completed", None)
    assert await asyncio.wait_for(shell.run("true"), 10) == ("[exit code: 0]", "completed", None)


@_LINUX_PROCESS_GROUPS
async def test_the_deadline_kills_the_commands_whole_process_group(shell):
    shell.timeout_seconds = 1
    workdir = Path(shell.workdir)
    started = asyncio.get_running_loop().time()

    text, status, error_type = await asyncio.wait_for(shell.run("(sleep 2; touch late.txt) & echo begun; sleep 5"), 10)

    assert text == "begun\n\n[command timed out after 1 s and was killed]"
    assert (status, error_type) == ("timeout", "timeout")
    assert asyncio.get_running_loop().time() - started < 4
    await asyncio.sleep(2.5)
    assert not (workdir / "late.txt").exists()


async def test_a_background_process_neither_holds_the_call_open_nor_dies_with_its_command(shell):
    workdir = Path(shell.workdir)

    # The background process keeps the command's output open; the call must still return at once.
    text, status, _ = await asyncio.wait_for(shell.run("(sleep 1.5; touch service.txt) & echo started"), 5)

    assert (text, status) == ("started\n\n[exit code: 0]", "completed")
    await _eventually(lambda: (workdir / "service.txt").exists())


async def test_long_output_keeps_its_head_and_tail(shell):
    text, _, _ = await asyncio.wait_for(
        shell.run("printf 'head'; head -c 5000 /dev/zero | tr '\\0' x; printf 'tail'"), 10
    )

    output = text.removesuffix("\n[exit code: 0]")
    assert output.startswith("head") and output.endswith("tail")
    assert "[... 4008 bytes of output omitted ...]" in output
    assert len(output) < 1100


async def test_a_command_that_deletes_the_session_directory_leaves_the_next_command_working(shell):
    await asyncio.wait_for(shell.run(f"rm -rf {shell.scratch}"), 10)

    assert await asyncio.wait_for(shell.run("echo still"), 10) == ("still\n\n[exit code: 0]", "completed", None)


@_LINUX_PROCESS_GROUPS
async def test_stopping_the_crew_kills_a_running_command(shell):
    workdir = Path(shell.workdir)
    running = asyncio.create_task(shell.run("touch started.txt; sleep 30; touch late.txt"))
    await _eventually(lambda: (workdir / "started.txt").exists())

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, 10)

    await asyncio.sleep(0.5)
    assert not (workdir / "late.txt").exists()


async def test_tool_runs_join_the_model_calls_that_asked_for_them():
    transcript = Transcript("rollout")
    transcript.tool_names = {"shell": "shell", "get_weather": "getWeather"}
    message = transcript.record_model_response(
        {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "Look it up.",
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
                            {"function": {"name": "shell", "arguments": {"command": "ls"}}},
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
        tool_name=lambda name: name,
    )

    async def ran(result):
        return result, "completed", None

    await transcript.run_tool("shell", {"command": "ls"}, lambda: ran("listing"))
    await transcript.run_tool("getWeather", {"city": "Paris"}, lambda: ran("sunny"))
    await transcript.run_tool("shell", {"command": "unasked"}, lambda: ran("stray"))
    report = transcript.report()

    # A call without an ID gets one, which CrewAI and the transcript share.
    shell_call_id = message["tool_calls"][1]["id"]
    assert shell_call_id.startswith("call_")
    assert [item["type"] for item in report["output"]] == ["reasoning", "function_call", "function_call"]
    # Verifiers see the tool's own name, not CrewAI's sanitized one.
    assert [item["name"] for item in report["output"][1:]] == ["getWeather", "shell"]
    assert json.loads(report["output"][2]["arguments"]) == {"command": "ls"}
    assert [(call["tool_call_id"], call["output"]) for call in report["tool_calls"]] == [
        ("c1", "sunny"),
        (shell_call_id, "listing"),
    ]
    assert report["response_ids"] == ["chatcmpl-1"]
    assert report["usage"]["total"] == 5
    assert {gap["code"] for gap in report["gaps"]} == {
        "tool_call_identity_unavailable",
        "cached_token_usage_unavailable",
        "reasoning_token_usage_unavailable",
    }


@pytest.mark.parametrize("missing", ["cached", "reasoning"])
def test_missing_detail_usage_does_not_become_a_known_zero_or_subtotal(missing):
    from responses_api_agents.crewai_agent.app import _usage

    transcript = Transcript("rollout")
    for number in (2, 4):
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": number},
            "completion_tokens_details": {"reasoning_tokens": number},
        }
        if number == 2:
            del usage["prompt_tokens_details" if missing == "cached" else "completion_tokens_details"]
        transcript._add_usage(usage)
    report = transcript.report()
    result = _usage(report["usage"])
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (20, 10, 30)
    assert result.input_tokens_details.cached_tokens == (None if missing == "cached" else 6)
    assert result.output_tokens_details.reasoning_tokens == (None if missing == "reasoning" else 6)
    assert any(gap["code"] == f"{missing}_token_usage_unavailable" for gap in report["gaps"])


@_LINUX_PROCESS_GROUPS
@pytest.mark.parametrize("cancel", [False, True])
async def test_deleted_scratch_cannot_disable_command_termination(shell, cancel):
    shell.timeout_seconds = 1
    workdir = Path(shell.workdir)
    running = asyncio.create_task(
        shell.run(f"rm -rf {shell.scratch}; touch started.txt; (sleep 2; touch late.txt) & sleep 10")
    )
    await _eventually(lambda: (workdir / "started.txt").exists())
    if cancel:
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, 5)
    else:
        _, status, error_type = await asyncio.wait_for(running, 5)
        assert (status, error_type) == ("timeout", "timeout")
    await asyncio.sleep(2.2)
    assert not (workdir / "late.txt").exists()


@pytest.mark.parametrize(
    "body, expected",
    [
        ('{"error": {"code": "context_length_exceeded", "message": "limit"}}', True),
        ('{"error": {"message": "context_length_exceeded"}}', False),
        ('{"error": {"code": "invalid_request_error"}}', False),
        ('{"error": "context_length_exceeded"}', False),
        ("context_length_exceeded", False),
    ],
)
def test_context_limit_requires_the_explicit_provider_code(body, expected):
    from responses_api_agents.crewai_agent.sandbox_runner import ModelRequestError

    error = ModelRequestError(400, body)
    assert error.context_limit is expected
    assert error.status == 400


@pytest.mark.parametrize(
    "status, body",
    [
        (403, '{"error": {"message": "Rate limit exceeded for this API key"}}'),
        (429, '{"error": {"message": "Too many requests"}}'),
    ],
)
@pytest.mark.parametrize("blocking", [False, True])
async def test_crewai_sends_a_failed_model_call_once(status, body, blocking):
    """The runner's model call is the only retry layer: CrewAI adds no attempts, even for rate-limit wording."""
    from responses_api_agents.crewai_agent.crew import CrewRun, GymChatLLM
    from responses_api_agents.crewai_agent.sandbox_runner import ModelRequestError

    calls = 0

    async def complete(messages, tools):
        nonlocal calls
        calls += 1
        raise ModelRequestError(status, body)

    llm = GymChatLLM(crew_run=CrewRun(asyncio.get_running_loop()), complete=complete, model="gym")
    with pytest.raises(ModelRequestError):
        if blocking:
            await asyncio.to_thread(llm.call, "task")
        else:
            await llm.acall("task")
    assert calls == 1
