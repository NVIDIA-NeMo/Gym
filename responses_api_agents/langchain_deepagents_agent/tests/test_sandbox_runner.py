# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The in-sandbox runner against a stub Model Server, in a temporary working directory."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from deepagents.backends import LocalShellBackend
from langchain_core.messages import HumanMessage, ToolMessage, messages_from_dict, messages_to_dict

from responses_api_agents.langchain_deepagents_agent import sandbox_runner
from responses_api_agents.langchain_deepagents_agent.sandbox_runner import TaskShellBackend
from responses_api_agents.langchain_deepagents_agent.tests.fakes import ModelServer, call, text


@pytest.fixture
def workdir(tmp_path, monkeypatch) -> Path:
    """The task's working directory; the runner changes into it, and the test's directory is restored after."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "session").mkdir()
    workdir = tmp_path / "task"
    workdir.mkdir()
    (workdir / "input.txt").write_text("from the task image\n")
    return workdir


def _payload(model: ModelServer, workdir: Path, **overrides) -> dict:
    return {
        "messages": messages_to_dict([HumanMessage(content="Do the task.")]),
        "model_url": f"{model.base_url}/responses",
        "request_defaults": {"temperature": 0.5},
        "max_input_tokens": 100_000,
        "system_prompt": None,
        "shell_timeout_seconds": 30,
        "workdir": str(workdir),
        "task_env": {"TASK_VAR": "from-the-owner"},
        "rollout_id": "rollout",
        "stop_request_path": str(workdir.parent / "session" / "stop.request"),
        **overrides,
    }


def _shell(workdir: Path) -> TaskShellBackend:
    return TaskShellBackend(root_dir=str(workdir), virtual_mode=False, inherit_env=True, scratch=workdir.parent)


def _tool_outputs(output: dict) -> dict[str, str]:
    return {m.tool_call_id: m.content for m in messages_from_dict(output["messages"]) if isinstance(m, ToolMessage)}


async def _until_gone(pid: int) -> None:
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"process {pid} is still running")


async def test_the_graph_runs_its_shell_and_file_tools_in_the_working_directory(workdir):
    model = ModelServer(
        [
            [
                call("execute", {"command": "pwd; echo $TASK_VAR; cp input.txt copied.txt"}, "c1"),
                call("write_file", {"file_path": str(workdir / "note.txt"), "content": "written"}, "c2"),
                call("read_file", {"file_path": str(workdir / "input.txt")}, "c3"),
            ],
            [text("done")],
        ]
    )
    try:
        output = await sandbox_runner.run(_payload(model, workdir), workdir.parent / "session")
    finally:
        model.close()

    assert (output["status"], output["stop_reason"]) == ("completed", None)
    outputs = _tool_outputs(output)
    cwd, task_var, *_ = outputs["c1"].splitlines()
    assert (Path(cwd).resolve(), task_var) == (workdir.resolve(), "from-the-owner")
    assert "from the task image" in outputs["c3"]
    assert (workdir / "copied.txt").read_text() == "from the task image\n"
    assert (workdir / "note.txt").read_text() == "written"
    # Every model call takes the rollout's URL with the activation's sampling, and the graph offers its shell.
    assert model.paths == ["/ng-rollout/rollout/v1/responses"] * 2
    assert {request["temperature"] for request in model.requests} == {0.5}
    assert "execute" in {tool["name"] for tool in model.requests[0]["tools"]}
    assert (output["model_calls"], len(output["usages"])) == (2, 2)
    assert messages_from_dict(output["messages"])[-1].content == "done"
    assert sorted(path.name for path in workdir.iterdir()) == ["copied.txt", "input.txt", "note.txt"]


async def test_deepagents_keeps_an_evicted_tool_result_in_graph_state(workdir):
    # Larger than deepagents' eviction threshold, so the graph moves the result to a file of its own.
    model = ModelServer(
        [[call("execute", {"command": "head -c 300000 /dev/zero | tr '\\0' x"}, "c1")], [text("done")]]
    )
    try:
        output = await sandbox_runner.run(_payload(model, workdir), workdir.parent / "session")
    finally:
        model.close()

    assert output["status"] == "completed"
    assert sandbox_runner.HARNESS_FILES_ROOT in _tool_outputs(output)["c1"]
    assert sorted(path.name for path in workdir.iterdir()) == ["input.txt"]
    assert not Path("/.deepagents").exists()


@pytest.mark.parametrize(
    ("answers", "status", "requests"),
    [
        ([429, 503, [text("done")]], None, 3),
        ([408, 425, [text("done")]], None, 3),
        ([501, 599, [text("done")]], None, 3),
        *(([code], code, 1) for code in (400, 401, 403, 404, 413, 422)),
        ([429, 429, 429], 429, 3),
        ([502, 504, 500], 500, 3),
    ],
)
async def test_retryable_statuses_are_sent_again_and_others_fail_the_run(
    workdir, monkeypatch, answers, status, requests
):
    delays = []
    retry_delay = sandbox_runner._retry_delay
    # Record each wait the runner asks for, without sleeping it.
    monkeypatch.setattr(sandbox_runner, "_retry_delay", lambda *args: delays.append(retry_delay(*args)) or 0)
    model = ModelServer(answers)
    try:
        output = await sandbox_runner.run(_payload(model, workdir), workdir.parent / "session")
    finally:
        model.close()

    assert len(model.requests) == requests
    assert len(delays) == requests - 1
    assert sum(delays) <= 2 * sandbox_runner._MAX_RETRY_DELAY_S
    assert output["model_calls"] == 1
    if status is None:
        assert output["status"] == "completed"
    else:
        assert (output["status"], output["model_error_status"]) == ("failed", status)
        assert f"HTTP {status}" in output["model_error"]


async def test_an_unreachable_model_server_is_a_transport_failure(workdir, monkeypatch):
    monkeypatch.setattr(sandbox_runner, "_retry_delay", lambda retry_after, attempt: 0)
    model = ModelServer([])
    model.close()

    output = await sandbox_runner.run(_payload(model, workdir), workdir.parent / "session")

    assert (output["status"], output["model_transport_error"]) == ("failed", True)


def test_the_retry_delay_follows_retry_after_within_its_cap():
    assert sandbox_runner._retry_delay("2", 1) == 2.0
    assert sandbox_runner._retry_delay(None, 3) == 8.0
    assert sandbox_runner._retry_delay("soon", 1) == 2.0
    assert sandbox_runner._retry_delay("600", 1) == 30.0


async def test_shell_output_reads_as_local_shell_backends(workdir):
    command = "echo out; echo err1 >&2; echo err2 >&2; exit 3"
    ours = await _shell(workdir).aexecute(command)
    theirs = LocalShellBackend(root_dir=str(workdir), virtual_mode=False, inherit_env=True).execute(command)

    assert (ours.output, ours.exit_code, ours.truncated) == (theirs.output, theirs.exit_code, theirs.truncated)
    assert ours.exit_code == 3


async def test_a_background_process_does_not_hold_a_command_open(workdir):
    result = await asyncio.wait_for(_shell(workdir).aexecute("sleep 30 & echo $! > child.pid; echo started"), 10)
    try:
        assert (result.output, result.exit_code) == ("started\n", 0)
    finally:
        os.kill(int((workdir / "child.pid").read_text()), signal.SIGKILL)


async def test_a_command_past_its_timeout_is_killed_with_its_process_group(workdir):
    result = await asyncio.wait_for(
        _shell(workdir).aexecute("sleep 30 & echo $! > child.pid; sleep 30", timeout=1), 10
    )

    assert result.exit_code == 124
    assert "timed out after 1 seconds (custom timeout)" in result.output
    await _until_gone(int((workdir / "child.pid").read_text()))


async def test_a_stopped_runner_kills_the_command_it_was_running(workdir):
    running = asyncio.create_task(_shell(workdir).aexecute("echo $$ > shell.pid; sleep 30"))
    for _ in range(100):
        if (workdir / "shell.pid").exists() and (workdir / "shell.pid").read_text():
            break
        await asyncio.sleep(0.05)
    running.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, 10)
    await _until_gone(int((workdir / "shell.pid").read_text()))


@pytest.mark.parametrize(("stop_requested", "stop_reason"), [(False, "wall_time"), (True, "cancelled")])
def test_sigterm_reports_the_messages_finished_so_far(workdir, stop_requested, stop_reason):
    model = ModelServer([[call("execute", {"command": "echo first"}, "c1")], None])
    session = workdir.parent / "session"
    input_path, output_path = session / "input.json", session / "output.json"
    payload = _payload(model, workdir)
    input_path.write_text(json.dumps(payload))
    if stop_requested:
        Path(payload["stop_request_path"]).write_text("")
    process = subprocess.Popen([sys.executable, sandbox_runner.__file__, str(input_path), str(output_path)])
    try:
        assert model.blocked.wait(60), "the runner never made its second model call"
        process.send_signal(signal.SIGTERM)
        assert process.wait(30) == 0
    finally:
        if process.poll() is None:
            process.kill()
        model.close()

    output = json.loads(output_path.read_text())
    assert (output["status"], output["stop_reason"]) == ("incomplete", stop_reason)
    assert "first" in _tool_outputs(output)["c1"]
    # The second call never returned, so it has no usage.
    assert (output["model_calls"], len(output["usages"])) == (2, 1)


def test_a_runner_that_fails_reports_its_error(tmp_path):
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps({"workdir": str(tmp_path / "missing")}))

    sandbox_runner.main(str(input_path), str(output_path))

    output = json.loads(output_path.read_text())
    assert output["status"] == "failed"
    assert "FileNotFoundError" in output["error"]
    assert "Traceback" in output["traceback"]


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
    from responses_api_agents.langchain_deepagents_agent.sandbox_runner import ModelRequestError

    error = ModelRequestError(400, body)
    assert error.context_limit is expected
    assert error.status == 400


async def test_explicit_context_rejection_is_a_gradable_context_stop(workdir):
    model = ModelServer([(400, {"error": {"code": "context_length_exceeded", "message": "limit"}})])
    try:
        output = await sandbox_runner.run(_payload(model, workdir), workdir.parent / "session")
    finally:
        model.close()
    assert len(model.requests) == 1
    assert (output["status"], output["stop_reason"]) == ("incomplete", "context_limit")
    assert output["model_context_limit"] is True
    assert output["model_error"] is None


async def test_deleted_scratch_is_recreated_before_the_next_command(workdir):
    scratch = workdir.parent / "command-scratch"
    scratch.mkdir()
    shell = TaskShellBackend(root_dir=str(workdir), virtual_mode=False, inherit_env=True, scratch=scratch)
    removed = await shell.aexecute(f"rm -rf {scratch}")
    assert removed.exit_code == 0
    result = await shell.aexecute("echo recovered")
    assert result.exit_code == 0
    assert result.output == "recovered\n"


async def test_success_body_interruption_keeps_usage_unknown_after_recovery(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import aiohttp

    from responses_api_agents.langchain_deepagents_agent.app import _usage
    from responses_api_agents.langchain_deepagents_agent.tests.fakes import USAGE

    class Reply:
        status = 200
        headers = {}

        def __init__(self, read):
            self.json = read

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    replies = iter(
        [
            Reply(AsyncMock(side_effect=aiohttp.ClientPayloadError("truncated"))),
            Reply(AsyncMock(return_value={"usage": USAGE})),
            Reply(AsyncMock(return_value={"usage": USAGE})),
        ]
    )
    model = sandbox_runner.SandboxResponsesChatModel(
        http=SimpleNamespace(post=lambda *args, **kwargs: next(replies)), url="http://model/responses"
    )
    monkeypatch.setattr(sandbox_runner, "_retry_delay", lambda *args: 0)
    for _ in range(2):
        await model.create_response({})
        assert (
            _usage({"model_calls": model.calls, "usages": model.usages, "usage_incomplete": model.usage_incomplete})
            is None
        )
    assert model.calls == 2
    assert model.usages == [USAGE, USAGE]


def test_a_runner_stopped_before_its_graph_starts_reports_an_empty_stop(workdir, monkeypatch):
    model = ModelServer([[text("never")]])
    session = workdir.parent / "session"
    input_path, output_path = session / "input.json", session / "output.json"
    input_path.write_text(json.dumps(_payload(model, workdir)))
    # As if SIGTERM arrived while the runner was still importing deepagents.
    monkeypatch.setattr(sandbox_runner, "_terminated_before_start", True)
    try:
        sandbox_runner.main(str(input_path), str(output_path))
    finally:
        model.close()

    output = json.loads(output_path.read_text())
    assert (output["status"], output["stop_reason"], output["messages"]) == ("incomplete", "wall_time", [])
    assert (output["model_calls"], model.requests) == (0, [])


def test_a_sigterm_while_the_runner_starts_still_leaves_a_report(workdir):
    model = ModelServer([None])
    session = workdir.parent / "session"
    input_path, output_path = session / "input.json", session / "output.json"
    input_path.write_text(json.dumps(_payload(model, workdir)))
    process = subprocess.Popen([sys.executable, sandbox_runner.__file__, str(input_path), str(output_path)])
    try:
        # Usually still importing deepagents; a later signal must leave a report too.
        time.sleep(0.5)
        process.send_signal(signal.SIGTERM)
        assert process.wait(60) == 0
    finally:
        if process.poll() is None:
            process.kill()
        model.close()

    output = json.loads(output_path.read_text())
    assert (output["status"], output["stop_reason"]) == ("incomplete", "wall_time")


def test_the_report_is_written_without_waiting_for_file_tool_threads(workdir, monkeypatch):
    async def run_leaving_a_thread(payload: dict, scratch: Path) -> dict:
        # deepagents runs file tools in threads that its own timeouts do not stop.
        asyncio.get_running_loop().run_in_executor(None, time.sleep, 5)
        return {"status": "completed", "stop_reason": None, "model_error": None, "messages": []}

    monkeypatch.setattr(sandbox_runner, "run", run_leaving_a_thread)
    session = workdir.parent / "session"
    input_path, output_path = session / "input.json", session / "output.json"
    input_path.write_text(json.dumps({"stop_request_path": str(session / "stop.request")}))

    started = time.monotonic()
    sandbox_runner.main(str(input_path), str(output_path))

    assert time.monotonic() - started < 3
    assert json.loads(output_path.read_text())["status"] == "completed"
