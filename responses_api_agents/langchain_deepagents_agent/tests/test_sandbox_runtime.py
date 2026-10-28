# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real DeepAgents loop and shell with a scripted model transport."""

import json
import os
import signal
import time
from uuid import uuid4

import pytest

from responses_api_agents.langchain_deepagents_agent import sandbox_runner
from responses_api_agents.langchain_deepagents_agent.runtime_observer import DeepAgentsObserver


class Reply:
    def __init__(self, status, body):
        self.status, self.body = status, body
        self.headers = {"Retry-After": "0"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def json(self):
        return self.body

    async def text(self):
        return json.dumps(self.body)


class Transport:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def post(self, url, *, json, headers):
        self.requests.append({"url": url, "json": json, "headers": headers})
        return next(self.replies)


def response(items, index, usage=True):
    return {
        "id": f"response-{index}",
        "output": items,
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12} if usage else None,
    }


def payload(tmp_path):
    return {
        "input": "Run the requested shell command",
        "workdir": str(tmp_path),
        "shell_timeout_seconds": 5,
        "model_url": "http://model/ng-rollout/rollout/v1/responses",
        "model_defaults": {"temperature": 1, "top_p": 0.95, "max_output_tokens": 32768},
        "assistant_message_header": "x-deepagents-model-run-id",
        "max_input_tokens": 262144,
        "recursion_limit": 30,
        "stop_request_path": str(tmp_path / "stop"),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code,usage", [(0, True), (7, True), (0, False)])
async def test_native_loop_retains_actual_tool_and_model_evidence(tmp_path, monkeypatch, exit_code, usage):
    marker = tmp_path / "marker"
    command = f"printf expected > '{marker}'; printf observed; exit {exit_code}"
    transport = Transport(
        [
            Reply(
                200,
                response(
                    [
                        {
                            "type": "function_call",
                            "name": "execute",
                            "call_id": "call-1",
                            "arguments": json.dumps({"command": command}),
                        }
                    ],
                    1,
                    usage,
                ),
            ),
            Reply(
                200,
                response(
                    [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}],
                    2,
                    usage,
                ),
            ),
        ]
    )
    monkeypatch.setattr(sandbox_runner, "model_http_session", lambda: transport)
    observer = DeepAgentsObserver(
        output=tmp_path / "output.json",
        task_id="task",
        rollout_id="rollout",
        model_ref={"type": "responses_api_models", "name": "policy_model"},
        sandbox_id="sandbox",
    )
    await sandbox_runner.run(payload(tmp_path), observer)
    assert observer.status == "completed", observer.error
    assert marker.read_text() == "expected"
    assert len(observer.turns) == 2
    assert [ref["response_id"] for ref in observer.invocations["root"]["model_calls"]] == ["response-1", "response-2"]
    (tool,) = observer.tools.values()
    assert tool["tool_call_id"] == "call-1"
    assert tool["status"] == ("failed" if exit_code else "completed")
    assert tool["completed_at"] >= tool["started_at"]
    results = [i for i in transport.requests[1]["json"]["input"] if i["type"] == "function_call_output"]
    assert len(results) == 1 and "observed" in results[0]["output"]
    assert all(r["url"] == payload(tmp_path)["model_url"] for r in transport.requests)
    assert all(r["headers"]["x-session-id"] == "root" for r in transport.requests)
    assert [r["headers"]["x-deepagents-model-run-id"] for r in transport.requests] == [
        t["source_message_id"] for t in observer.turns
    ]
    assert all(u is not None for u in observer.usages) == usage
    retained = json.loads((tmp_path / "output.json").read_text())
    assert any(i.get("call_id") == "call-1" and i["type"] == "function_call_output" for i in retained["output_items"])
    assert transport.requests[0]["json"]["max_output_tokens"] == 32768


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 500])
async def test_model_retries_keep_request_and_native_turn_identity(tmp_path, monkeypatch, status):
    transport = Transport(
        [
            Reply(status, {"error": "retry"}),
            Reply(
                200,
                response(
                    [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}], 1
                ),
            ),
        ]
    )
    monkeypatch.setattr(sandbox_runner, "model_http_session", lambda: transport)
    observer = DeepAgentsObserver(
        output=tmp_path / "output.json",
        task_id="task",
        rollout_id="rollout",
        model_ref={"type": "responses_api_models", "name": "policy_model"},
        sandbox_id="sandbox",
    )
    await sandbox_runner.run(payload(tmp_path), observer)
    assert observer.status == "completed", observer.error
    assert len(observer.turns) == 1
    assert len(transport.requests) == 2
    assert transport.requests[0] == transport.requests[1]


@pytest.mark.asyncio
async def test_model_error_retains_partial_evidence_and_original_status(tmp_path, monkeypatch):
    transport = Transport([Reply(400, {"error": "preserved rejection"})])
    monkeypatch.setattr(sandbox_runner, "model_http_session", lambda: transport)
    observer = DeepAgentsObserver(
        output=tmp_path / "output.json",
        task_id="task",
        rollout_id="rollout",
        model_ref={"type": "responses_api_models", "name": "policy_model"},
        sandbox_id="sandbox",
    )
    await sandbox_runner.run(payload(tmp_path), observer)
    assert observer.status == "failed"
    assert observer.error_status == 400 and "preserved rejection" in observer.error
    assert len(observer.turns) == len(transport.requests) == 1
    assert json.loads((tmp_path / "output.json").read_text())["status"] == "failed"


def test_background_process_does_not_hold_tool_stdout_open(tmp_path):
    backend = sandbox_runner.SupervisedShellBackend(workdir=str(tmp_path), scratch=tmp_path, timeout=4)
    pidfile = tmp_path / "pid"
    started = time.monotonic()
    pid = None
    try:
        result = backend.execute(f"sleep 30 & echo $! > '{pidfile}'; printf returned")
        pid = int(pidfile.read_text())
        assert result.exit_code == 0 and result.output == "returned"
        assert time.monotonic() - started < 3
    finally:
        if pid:
            os.kill(pid, signal.SIGKILL)


def test_shell_timeout_and_non_utf8_are_reported(tmp_path):
    backend = sandbox_runner.SupervisedShellBackend(workdir=str(tmp_path), scratch=tmp_path, timeout=1)
    result = backend.execute("printf '\\377'; sleep 10")
    assert result.exit_code == 124 and "\ufffd" in result.output
    assert "timed out" in result.output


@pytest.fixture(autouse=True)
def restore_process_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)


@pytest.mark.asyncio
async def test_native_subagent_models_and_tools_have_distinct_ownership(tmp_path, monkeypatch):
    def text(value):
        return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": value}]}

    transport = Transport(
        [
            Reply(
                200,
                response(
                    [
                        {
                            "type": "function_call",
                            "name": "task",
                            "call_id": "delegate",
                            "arguments": json.dumps(
                                {"description": "Run the shell check", "subagent_type": "general-purpose"}
                            ),
                        }
                    ],
                    1,
                ),
            ),
            Reply(
                200,
                response(
                    [
                        {
                            "type": "function_call",
                            "name": "execute",
                            "call_id": "child-shell",
                            "arguments": json.dumps({"command": "printf child"}),
                        }
                    ],
                    2,
                ),
            ),
            Reply(200, response([text("child done")], 3)),
            Reply(200, response([text("all done")], 4)),
        ]
    )
    monkeypatch.setattr(sandbox_runner, "model_http_session", lambda: transport)
    observer = DeepAgentsObserver(
        output=tmp_path / "output.json",
        task_id="task",
        rollout_id="rollout",
        model_ref={"type": "responses_api_models", "name": "policy_model"},
        sandbox_id="sandbox",
    )
    await sandbox_runner.run(payload(tmp_path), observer)
    assert observer.status == "completed", observer.error
    assert set(observer.invocations) == {"root", "root/delegate"}
    child = observer.invocations["root/delegate"]
    assert child["parent_invocation_id"] == "root" and child["spawned_by_tool_call_id"] == "delegate"
    assert [r["headers"]["x-session-id"] for r in transport.requests] == [
        "root",
        "root/delegate",
        "root/delegate",
        "root",
    ]
    assert child["status"] == "completed"
    tools = {t["tool_call_id"]: t for t in observer.tools.values()}
    assert tools["child-shell"]["invocation_id"] == "root/delegate"
    assert tools["delegate"]["invocation_id"] == "root"
    assert not observer.gaps
    outputs = [item for item in observer.invocations["root"]["conversation"] if item["type"] == "function_call_output"]
    assert len(outputs) == 1 and outputs[0]["call_id"] == "delegate"
    assert "child done" in outputs[0]["output"]


@pytest.mark.asyncio
async def test_native_compaction_is_captured_and_history_is_private(tmp_path, monkeypatch):
    def text(value):
        return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": value}]}

    transport = Transport(
        [Reply(200, response([text("retained summary")], 1)), Reply(200, response([text("final answer")], 2))]
    )
    monkeypatch.setattr(sandbox_runner, "model_http_session", lambda: transport)
    observer = DeepAgentsObserver(
        output=tmp_path / "output.json",
        task_id="task",
        rollout_id="rollout",
        model_ref={"type": "responses_api_models", "name": "policy_model"},
        sandbox_id="sandbox",
    )
    params = payload(tmp_path)
    params["max_input_tokens"] = 2000
    params["input"] = [
        {
            "type": "message",
            "role": ("user" if i % 2 == 0 else "assistant"),
            "content": f"turn {i}: " + "retained context " * 100,
        }
        for i in range(20)
    ]
    await sandbox_runner.run(params, observer)
    assert observer.status == "completed", observer.error
    assert len(transport.requests) == 2
    assert len(observer.compactions) == 1
    assert len(observer.turns) == 1
    compaction = observer.compactions[0]
    assert compaction["outcome"] == "completed" and compaction["summary"] == "retained summary"
    assert compaction["source_message_ids"] == [transport.requests[0]["headers"]["x-deepagents-model-run-id"]]
    assert len(observer.usages) == 2
    history = tmp_path / "conversation-history" / "rollout.md"
    assert history.is_file() and "turn 0" in history.read_text()


def test_shell_empty_command_preserves_native_tool_error(tmp_path):
    backend = sandbox_runner.SupervisedShellBackend(workdir=str(tmp_path), scratch=tmp_path, timeout=5)
    result = backend.execute("")
    assert result.exit_code == 1 and result.output == "Error: Command must be a non-empty string."


def test_shell_launch_failure_remains_recoverable(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    backend = sandbox_runner.SupervisedShellBackend(workdir=str(workdir), scratch=tmp_path, timeout=5)
    workdir.rmdir()
    failed = backend.execute("printf recovered")
    assert failed.exit_code == 1 and "FileNotFoundError" in failed.output
    workdir.mkdir()
    recovered = backend.execute("printf recovered")
    assert recovered.exit_code == 0 and recovered.output == "recovered"


@pytest.mark.asyncio
async def test_native_state_updating_tool_retains_its_actual_message(tmp_path, monkeypatch):
    transport = Transport(
        [
            Reply(
                200,
                response(
                    [
                        {
                            "type": "function_call",
                            "name": "write_todos",
                            "call_id": "todo-1",
                            "arguments": json.dumps(
                                {"todos": [{"content": "Inspect evidence", "status": "completed"}]}
                            ),
                        }
                    ],
                    1,
                ),
            ),
            Reply(
                200,
                response(
                    [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "done"}],
                        }
                    ],
                    2,
                ),
            ),
        ]
    )
    monkeypatch.setattr(sandbox_runner, "model_http_session", lambda: transport)
    observer = DeepAgentsObserver(
        output=tmp_path / "output.json",
        task_id="task",
        rollout_id="rollout",
        model_ref={"type": "responses_api_models", "name": "policy_model"},
        sandbox_id="sandbox",
    )
    await sandbox_runner.run(payload(tmp_path), observer)
    assert observer.status == "completed", observer.error
    assert not observer.gaps
    (tool,) = observer.tools.values()
    assert tool["tool_name"] == "write_todos" and tool["status"] == "completed"
    retained = [
        item for item in observer.invocations["root"]["conversation"] if item["type"] == "function_call_output"
    ]
    sent = [item for item in transport.requests[1]["json"]["input"] if item["type"] == "function_call_output"]
    assert retained == sent and len(retained) == 1
    assert retained[0]["call_id"] == "todo-1" and "Inspect evidence" in retained[0]["output"]


@pytest.mark.asyncio
@pytest.mark.parametrize("child", [False, True])
async def test_native_validation_error_retains_actual_repair_message(tmp_path, monkeypatch, child):
    def call(name, call_id, arguments):
        return {"type": "function_call", "name": name, "call_id": call_id, "arguments": json.dumps(arguments)}

    def text(value):
        return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": value}]}

    outputs = [
        [call("write_todos", "bad-todo", {"todos": [{"parameter": "Inspect evidence", "status": "completed"}]})],
        [call("write_todos", "fixed-todo", {"todos": [{"content": "Inspect evidence", "status": "completed"}]})],
        [text("recovered")],
    ]
    if child:
        outputs.insert(
            0, [call("task", "delegate", {"description": "Inspect evidence", "subagent_type": "general-purpose"})]
        )
        outputs.append([text("all done")])
    transport = Transport([Reply(200, response(items, index)) for index, items in enumerate(outputs)])
    monkeypatch.setattr(sandbox_runner, "model_http_session", lambda: transport)
    observer = DeepAgentsObserver(
        output=tmp_path / "output.json",
        task_id="task",
        rollout_id="rollout",
        model_ref={"type": "responses_api_models", "name": "policy_model"},
        sandbox_id="sandbox",
    )
    await sandbox_runner.run(payload(tmp_path), observer)
    assert observer.status == "completed", observer.error
    owner = "root/delegate" if child else "root"
    tool = next(t for t in observer.tools.values() if t["tool_call_id"] == "bad-todo")
    assert tool["status"] == "failed" and tool["error_type"] == "ValidationError"
    assert tool["invocation_id"] == owner and tool["completed_at"] >= tool["started_at"]
    sent = [
        item
        for item in transport.requests[2 if child else 1]["json"]["input"]
        if item["type"] == "function_call_output" and item["call_id"] == "bad-todo"
    ]
    assert len(sent) == 1 and "Field required" in sent[0]["output"]
    retained = [
        item
        for item in observer.invocations[owner]["conversation"]
        if item["type"] == "function_call_output" and item["call_id"] == "bad-todo"
    ]
    assert retained == sent
    assert not observer.gaps
    saved = json.loads((tmp_path / "output.json").read_text())
    saved_owner = next(i for i in saved["invocations"] if i["invocation_id"] == owner)
    assert saved_owner["conversation"] == observer.invocations[owner]["conversation"]
    if child:
        assert not any(item.get("call_id") == "bad-todo" for item in saved["output_items"])
    else:
        assert sent[0] in saved["output_items"]
    # The SDK sees the original invalid arguments, then its own error, and recovers.
    assert len(transport.requests) == len(outputs)
    assert all(r["url"] == payload(tmp_path)["model_url"] for r in transport.requests)


@pytest.mark.asyncio
async def test_tool_error_without_native_result_preserves_evidence_gap(tmp_path):
    observer = DeepAgentsObserver(
        output=tmp_path / "output.json",
        task_id="task",
        rollout_id="rollout",
        model_ref={"type": "responses_api_models", "name": "policy_model"},
        sandbox_id="sandbox",
    )
    run_id = uuid4()
    await observer.on_tool_start({"name": "write_todos"}, "{}", run_id=run_id, tool_call_id="unobserved")
    await observer.on_tool_error(ValueError("not a native tool result"), run_id=run_id)
    saved = json.loads((tmp_path / "output.json").read_text())
    assert saved["tools"][0]["status"] == "failed"
    assert saved["output_items"] == []
    assert saved["gaps"] == [{"code": "tool_output_unavailable", "invocation_id": "root", "detail": "unobserved"}]
