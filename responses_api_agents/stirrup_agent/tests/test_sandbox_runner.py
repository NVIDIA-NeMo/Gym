# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The in-sandbox runner against a stub model server and a resources app served over HTTP."""

import ast
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import stirrup.core.agent as stirrup_agent
from stirrup.clients.utils import to_openai_tools

from responses_api_agents.stirrup_agent import sandbox_runner
from responses_api_agents.stirrup_agent.nemo_agent import NeMoAgent
from responses_api_agents.stirrup_agent.sandbox import HARNESS_FILES
from responses_api_agents.stirrup_agent.sandbox_runner import LocalShell, Observer, main, remote_tool, run
from responses_api_agents.stirrup_agent.tests.fakes import _TOOLS, _call, _ModelServer, _Resources, _resources_app


@pytest.fixture(autouse=True)
def restore_run_tool(monkeypatch):
    """The runner replaces ``run_tool`` on Stirrup's agent classes for its process; undo that after each test."""
    for cls in (stirrup_agent.Agent, NeMoAgent):
        monkeypatch.setattr(cls, "run_tool", cls.run_tool)


@pytest.fixture
def calls() -> list:
    return []


@pytest.fixture
def resources(calls):
    server = _Resources(_resources_app(calls))
    yield server
    server.close()


def _payload(model_url: str, resources_url: str, workdir: Path) -> dict:
    return {
        "input": [{"role": "user", "content": "Write the report."}],
        "instructions": None,
        "tools": _TOOLS,
        "finish_tool_names": ["finish", "abandon_task_finish"],
        "code_exec": True,
        "tool_access": {"base_url": resources_url, "cookies": {"session": "seeded"}, "headers": {}},
        "workdir": str(workdir),
        "max_turns": 10,
        "min_compaction_summary_words": 1,
        "client": {"model": "policy", "base_url": model_url, "max_tokens": 32768, "temperature": 0.0},
    }


def test_a_conversation_runs_shell_commands_locally_and_task_tools_on_the_resources_server(tmp_path, calls, resources):
    workdir = tmp_path / "root"
    workdir.mkdir()
    model = _ModelServer(
        [
            _call("code_exec", json.dumps({"cmd": "echo draft > report.txt && pwd"})),
            _call("web_search", json.dumps({"query": "gdp"})),
            _call("web_search", json.dumps({"q": "gdp"})),
            _call("finish", json.dumps({"reason": "done", "paths": ["missing.txt"]})),
            _call("finish", json.dumps({"reason": "done", "paths": ["report.txt"]})),
        ]
    )
    try:
        output = asyncio.run(run(_payload(model.base_url, resources.base_url, workdir), tmp_path, Observer()))
    finally:
        model.close()

    assert (workdir / "report.txt").read_text() == "draft\n"
    results = [item["output"] for item in output["output_items"] if item["type"] == "function_call_output"]
    assert f"<stdout>{workdir}\n</stdout>" in results[0]
    assert results[1] == "results for gdp"
    assert results[2] == (
        "Tool arguments are not valid: query: Field required (type=missing). "
        'Submitted arguments (first 500 chars): \'{"q": "gdp"}\''
    )
    assert results[3] == "ERROR: Files do not exist: ['missing.txt']"
    assert [name for name, *_ in calls] == ["web_search", "finish", "finish"]
    assert calls[0][2] == {"session": "seeded"}
    assert calls[1][2] == {"session": "after-search"}
    assert output["resources_cookies"] == {"session": "after-search"}
    assert len(model.requests) == 5


def test_the_run_reports_its_model_calls_and_tool_executions(tmp_path, resources):
    workdir = tmp_path / "root"
    workdir.mkdir()
    model = _ModelServer(
        [
            _call("code_exec", json.dumps({"cmd": "true"})),
            _call("code_exec", json.dumps({"cmd": "exit 7"})),
            _call("web_search", json.dumps({"q": "gdp"})),
            _call("finish", json.dumps({"reason": "done"})),
        ]
    )
    try:
        output = asyncio.run(run(_payload(model.base_url, resources.base_url, workdir), tmp_path, Observer()))
    finally:
        model.close()

    observations = output["observations"]
    assert observations["status"] == "completed"
    assert observations["model_response_ids"] == ["chatcmpl-1", "chatcmpl-2", "chatcmpl-3", "chatcmpl-4"]
    assert model.session_ids == ["root"] * 4
    tools = observations["tool_calls"]
    assert [(t["tool_name"], t["status"]) for t in tools] == [
        ("code_exec", "completed"),
        ("code_exec", "failed"),
        ("web_search", "failed"),
        ("finish", "completed"),
    ]
    outputs = [item["call_id"] for item in output["output_items"] if item["type"] == "function_call_output"]
    assert [t["tool_call_id"] for t in tools] == outputs
    assert all(t["completed_at"] >= t["started_at"] > 1e9 and t["duration_ms"] >= 0 for t in tools)
    counts = [(u["prompt_tokens"], u["completion_tokens"], u["total_tokens"]) for u in output["usages"]]
    assert counts == [(10, 5, 15)] * 4


async def _cancelled_after_the_model_hangs(payload: dict, tmp_path: Path, model: _ModelServer) -> dict:
    task = asyncio.ensure_future(run(payload, tmp_path, Observer()))
    assert await asyncio.to_thread(model.blocked.wait, 30), "the model was never called"
    task.cancel()
    return await task


@pytest.mark.parametrize(("marker_exists", "stop_reason"), [(False, "wall_time"), (True, "cancelled")])
def test_a_cancelled_run_reports_the_turns_it_completed(tmp_path, resources, marker_exists, stop_reason):
    workdir = tmp_path / "root"
    workdir.mkdir()
    marker = tmp_path / "stop-requested"
    if marker_exists:
        marker.touch()
    model = _ModelServer([_call("code_exec", json.dumps({"cmd": "echo draft > report.txt"})), None])
    payload = {**_payload(model.base_url, resources.base_url, workdir), "stop_request_path": str(marker)}
    try:
        output = asyncio.run(_cancelled_after_the_model_hangs(payload, tmp_path, model))
    finally:
        model.close()

    assert output["stop_reason"] == stop_reason
    assert output["observations"]["status"] == "incomplete"
    assert [item["type"] for item in output["output_items"]].count("function_call") == 1
    assert [t["tool_name"] for t in output["observations"]["tool_calls"]] == ["code_exec"]
    assert (workdir / "report.txt").read_text() == "draft\n"


def test_sigterm_makes_the_runner_write_its_partial_report(tmp_path, resources):
    workdir = tmp_path / "root"
    workdir.mkdir()
    model = _ModelServer([_call("code_exec", json.dumps({"cmd": "echo draft > report.txt"})), None])
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps(_payload(model.base_url, resources.base_url, workdir)))
    repo = Path(__file__).resolve().parents[3]
    runner = repo / "responses_api_agents/stirrup_agent/sandbox_runner.py"
    process = subprocess.Popen(
        [sys.executable, str(runner), str(input_path), str(output_path)],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo)},
    )
    try:
        assert model.blocked.wait(60), "the model was never called"
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=60) == 0
    finally:
        process.kill()
        model.close()

    output = json.loads(output_path.read_text())
    assert (output["stop_reason"], output["observations"]["status"]) == ("wall_time", "incomplete")
    assert any(item["type"] == "function_call" for item in output["output_items"])
    assert (workdir / "report.txt").read_text() == "draft\n"


def _run_main(tmp_path: Path, payload: dict) -> dict:
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps(payload))
    main(str(input_path), str(output_path))
    return json.loads(output_path.read_text())


def test_task_tool_calls_retry_the_connection_before_the_request_is_sent(tmp_path, resources, monkeypatch):
    transports = []
    real_transport = httpx.AsyncHTTPTransport

    def recording_transport(**kwargs):
        transports.append(kwargs)
        return real_transport(**kwargs)

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", recording_transport)
    model = _ModelServer([_call("finish", json.dumps({"reason": "done", "paths": []}))])
    try:
        asyncio.run(run(_payload(model.base_url, resources.base_url, tmp_path), tmp_path, Observer()))
    finally:
        model.close()

    assert transports == [{"retries": 3}]


def test_a_failed_run_still_reports_what_it_observed(tmp_path, resources):
    model = _ModelServer(
        [_call("code_exec", json.dumps({"cmd": "true"})), _call("fetch_web_page", json.dumps({"url": "x"}))]
    )
    try:
        output = _run_main(tmp_path, _payload(model.base_url, resources.base_url, tmp_path))
    finally:
        model.close()

    assert "500" in output["error"]
    assert "Traceback" in output["traceback"]
    observations = output["observations"]
    assert observations["status"] == "failed"
    assert observations["model_response_ids"] == ["chatcmpl-1", "chatcmpl-2"]
    assert [t["tool_name"] for t in observations["tool_calls"]] == ["code_exec"]


@pytest.mark.parametrize("status", [400, 413, 422])
def test_a_request_the_model_server_rejects_ends_the_run_as_failed_with_the_turns_so_far(tmp_path, resources, status):
    model = _ModelServer([_call("code_exec", json.dumps({"cmd": "true"})), status])
    try:
        output = asyncio.run(run(_payload(model.base_url, resources.base_url, tmp_path), tmp_path, Observer()))
    finally:
        model.close()

    assert output["stop_reason"] == "model_error"
    assert str(status) in output["model_error"]
    observations = output["observations"]
    assert observations["status"] == "failed"
    assert observations["model_response_ids"] == ["chatcmpl-1"]
    assert [t["tool_name"] for t in observations["tool_calls"]] == ["code_exec"]
    assert [item["type"] for item in output["output_items"]] == ["function_call", "function_call_output"]
    assert [u["total_tokens"] for u in output["usages"]] == [15]


@pytest.mark.parametrize("status", [401, 403, 404])
def test_a_model_server_error_that_is_not_a_rejected_request_still_fails_the_run(tmp_path, resources, status):
    model = _ModelServer([status])
    try:
        output = _run_main(tmp_path, _payload(model.base_url, resources.base_url, tmp_path))
    finally:
        model.close()

    assert str(status) in output["error"]
    assert output["observations"]["status"] == "failed"
    assert "model_error" not in output


def test_a_timed_out_command_is_a_timeout_observation_and_the_model_sees_the_failure(tmp_path, resources, monkeypatch):
    init = LocalShell.__init__

    def short_timeout(self, *args):
        init(self, *args)
        self._shell_timeout = 1

    monkeypatch.setattr(sandbox_runner.LocalShell, "__init__", short_timeout)
    workdir = tmp_path / "root"
    workdir.mkdir()
    model = _ModelServer(
        [
            _call("code_exec", json.dumps({"cmd": "sleep 30"})),
            _call("code_exec", json.dumps({"cmd": "true"})),
            _call("finish", json.dumps({"reason": "done"})),
        ]
    )
    try:
        output = asyncio.run(run(_payload(model.base_url, resources.base_url, workdir), tmp_path, Observer()))
    finally:
        model.close()

    tools = output["observations"]["tool_calls"]
    assert [(t["tool_name"], t["status"]) for t in tools] == [
        ("code_exec", "timeout"),
        ("code_exec", "completed"),
        ("finish", "completed"),
    ]
    result = [item["output"] for item in output["output_items"] if item["type"] == "function_call_output"][0]
    assert "<exit_code>1</exit_code>" in result
    assert "Command timed out after 1 seconds" in result


def test_a_run_that_uses_all_its_turns_without_finishing_is_incomplete(tmp_path, resources):
    workdir = tmp_path / "root"
    workdir.mkdir()
    model = _ModelServer([_call("code_exec", json.dumps({"cmd": "true"}))] * 2)
    payload = {**_payload(model.base_url, resources.base_url, workdir), "max_turns": 2}
    try:
        output = asyncio.run(run(payload, tmp_path, Observer()))
    finally:
        model.close()

    assert (output["observations"]["status"], output["stop_reason"]) == ("incomplete", None)
    assert len(model.requests) == 2


def test_the_model_sees_the_certified_tool_definitions(tmp_path, resources):
    model = _ModelServer([_call("finish", json.dumps({"reason": "done"}))])
    try:
        asyncio.run(run(_payload(model.base_url, resources.base_url, tmp_path), tmp_path, Observer()))
    finally:
        model.close()

    tools = model.requests[0]["tools"]
    # The certified order: finish tools, then each provider's tools in turn.
    assert [t["function"]["name"] for t in tools] == [
        "finish",
        "abandon_task_finish",
        "code_exec",
        "web_search",
        "fetch_web_page",
    ]
    certified = {
        t["name"]: {"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters")}}
        for t in _TOOLS
    }
    assert [t for t in tools if t["function"]["name"] != "code_exec"] == [
        certified[name] for name in ("finish", "abandon_task_finish", "web_search", "fetch_web_page")
    ]


def test_without_code_exec_the_model_sees_only_the_request_tools(tmp_path, resources):
    model = _ModelServer([_call("finish", json.dumps({"reason": "done"}))])
    payload = {**_payload(model.base_url, resources.base_url, tmp_path), "code_exec": False}
    try:
        output = asyncio.run(run(payload, tmp_path, Observer()))
    finally:
        model.close()

    assert output["observations"]["status"] == "completed"
    assert [t["function"]["name"] for t in model.requests[0]["tools"]] == [
        "finish",
        "abandon_task_finish",
        "web_search",
        "fetch_web_page",
    ]


def test_remote_tool_schema_is_the_rows_schema():
    spec = next(t for t in _TOOLS if t["name"] == "finish")

    tool = remote_tool(spec, client=None)

    assert to_openai_tools({tool.name: tool})[0]["function"]["parameters"] == spec["parameters"]


@pytest.mark.parametrize("body", [{"found": True, "ids": [1, 2]}, [{"id": 1}], 7, None, "plain text"])
def test_a_route_reply_that_is_not_a_string_reaches_the_model_as_json(body):
    result = _call_remote_tool(lambda request: httpx.Response(200, content=json.dumps(body)))

    assert result.success is True
    assert result.content == (body if isinstance(body, str) else json.dumps(body))


@pytest.mark.parametrize("detail", [{"error": "no such file"}, ["a", "b"], "ERROR: Files do not exist"])
def test_a_refusal_detail_that_is_not_a_string_reaches_the_model_as_json(detail):
    result = _call_remote_tool(lambda request: httpx.Response(400, json={"detail": detail}))

    assert result.success is False
    assert result.content == (detail if isinstance(detail, str) else json.dumps(detail))


@pytest.mark.parametrize(
    "response", [httpx.Response(200, content=b""), httpx.Response(400, content=b"<html>bad gateway</html>")]
)
def test_a_route_reply_that_is_not_json_fails_the_run(response):
    with pytest.raises(json.JSONDecodeError):
        _call_remote_tool(lambda request: response)


def test_a_422_whose_detail_is_not_a_validation_error_list_reaches_the_model(tmp_path, resources):
    model = _ModelServer(
        [_call("finish", json.dumps({"reason": "unprocessable"})), _call("finish", '{"reason": "ok"}')]
    )
    try:
        output = asyncio.run(run(_payload(model.base_url, resources.base_url, tmp_path), tmp_path, Observer()))
    finally:
        model.close()

    results = [item["output"] for item in output["output_items"] if item["type"] == "function_call_output"]
    assert "reason is not accepted" in results[0]


def _call_remote_tool(handler):
    async def call():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://resources") as client:
            tool = remote_tool({"name": "lookup", "parameters": {"type": "object", "properties": {}}}, client)
            return await tool.executor(tool.parameters())

    return asyncio.run(call())


class TestLocalShell:
    @pytest.fixture
    def shell(self, tmp_path) -> LocalShell:
        (tmp_path / "root").mkdir()
        return LocalShell(str(tmp_path / "root"), tmp_path)

    def _run(self, shell: LocalShell, cmd: str, **kwargs):
        return asyncio.run(shell.run_command(cmd, **kwargs))

    def test_no_shell_state_carries_over(self, shell, tmp_path):
        self._run(shell, "cd /tmp; export X=1")

        result = self._run(shell, 'pwd; echo "${X:-unset}"')

        assert result.stdout == f"{tmp_path / 'root'}\nunset\n"

    def test_output_keeps_streams_apart_and_ends_in_a_newline(self, shell):
        result = self._run(shell, "printf out; echo err >&2; exit 3")

        assert (result.exit_code, result.stdout, result.stderr) == (3, "out\n", "err\n")

    def test_a_timeout_is_reported_after_stderr(self, shell):
        result = self._run(shell, "echo partial >&2; sleep 30", timeout=1)

        assert (result.exit_code, result.stderr) == (1, "partial\n\nCommand timed out after 1 seconds")

    def test_a_background_process_does_not_hold_the_call(self, shell):
        started = time.monotonic()

        result = self._run(shell, "sleep 20 & echo started")

        assert result.stdout == "started\n"
        assert time.monotonic() - started < 10


def test_the_staged_files_have_no_broken_imports():
    """Tests run with the whole repo importable, so a module missing from the sandbox would go unnoticed."""
    package = Path(__file__).resolve().parents[1]
    for name in HARNESS_FILES:
        tree = ast.parse((package / name).read_text())
        type_checking_only = {
            id(node)
            for block in ast.walk(tree)
            if isinstance(block, ast.If) and ast.unparse(block.test) == "TYPE_CHECKING"
            for node in ast.walk(block)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom | ast.Import) or id(node) in type_checking_only:
                continue
            for module in [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]:
                assert not module.startswith(("nemo_gym", "resources_servers")), f"{name} imports {module}"
                if module.startswith("responses_api_agents"):
                    assert f"{module.rsplit('.', 1)[-1]}.py" in HARNESS_FILES, f"{name} imports {module}"
