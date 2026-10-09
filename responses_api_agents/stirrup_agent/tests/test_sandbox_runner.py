# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The in-sandbox runner against a stub model server and a resources app served over HTTP."""

import ast
import asyncio
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import stirrup.core.agent as stirrup_agent
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from stirrup.clients.utils import to_openai_tools

from responses_api_agents.stirrup_agent.nemo_agent import NeMoAgent
from responses_api_agents.stirrup_agent.sandbox import HARNESS_FILES
from responses_api_agents.stirrup_agent.sandbox_runner import InvocationStatus, LocalShell, Observer, remote_tool, run


_TOOLS = json.loads((Path(__file__).resolve().parents[3] / "benchmarks" / "gdpval" / "tools.json").read_text())
_CERTIFIED = json.loads(
    (Path(__file__).resolve().parents[3] / "resources_servers/gdpval/tests/data/tool_definitions.json").read_text()
)


def _call(name: str, arguments: str) -> dict:
    tool_call = {"id": f"call-{name}", "type": "function", "function": {"name": name, "arguments": arguments}}
    return {"role": "assistant", "content": None, "tool_calls": [tool_call]}


class _ModelServer:
    """Answer chat completions in order and record each request body."""

    def __init__(self, answers: list[dict]) -> None:
        self.answers = answers
        self.requests: list[dict] = []
        self.session_ids: list[str | None] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                server.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                server.session_ids.append(self.headers.get("x-session-id"))
                message = server.answers[len(server.requests) - 1]
                payload = json.dumps(
                    {
                        "id": f"chatcmpl-{len(server.requests)}",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "policy",
                        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class _Finish(BaseModel):
    reason: str
    paths: list[str] = []


class _Search(BaseModel):
    query: str


def _resources_app(calls: list[tuple[str, dict, dict]]) -> FastAPI:
    """Routes that answer like the GDPVal resources server: a JSON string, a 400 detail or a 422."""
    app = FastAPI()

    @app.post("/web_search")
    async def web_search(body: _Search, request: Request, response: Response) -> str:
        calls.append(("web_search", body.model_dump(), dict(request.cookies)))
        response.set_cookie("session", "after-search")
        return f"results for {body.query}"

    @app.post("/finish")
    async def finish(body: _Finish, request: Request) -> str:
        calls.append(("finish", body.model_dump(), dict(request.cookies)))
        if "missing.txt" in body.paths:
            raise HTTPException(400, "ERROR: Files do not exist: ['missing.txt']")
        return body.reason

    @app.post("/fetch_web_page")
    async def fetch_web_page() -> str:
        raise HTTPException(500, "boom")

    return app


class _Resources:
    def __init__(self, app: FastAPI) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self._server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            assert time.monotonic() < deadline, "resources app did not start"
            time.sleep(0.01)

    def close(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


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
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert output["usages"] == [{**usage, "completion_tokens_details": None, "prompt_tokens_details": None}] * 4


def test_a_failed_run_still_reports_what_it_observed(tmp_path, resources):
    model = _ModelServer(
        [_call("code_exec", json.dumps({"cmd": "true"})), _call("fetch_web_page", json.dumps({"url": "x"}))]
    )
    observer = Observer()
    try:
        with pytest.raises(Exception, match="500"):
            asyncio.run(run(_payload(model.base_url, resources.base_url, tmp_path), tmp_path, observer))
    finally:
        model.close()

    report = observer.report(InvocationStatus.FAILED)["observations"]
    assert report["status"] == "failed"
    assert report["model_response_ids"] == ["chatcmpl-1", "chatcmpl-2"]
    assert [t["tool_name"] for t in report["tool_calls"]] == ["code_exec"]


def test_a_server_error_from_a_task_tool_fails_the_run(tmp_path, resources):
    model = _ModelServer([_call("fetch_web_page", json.dumps({"url": "https://example.com"}))])
    try:
        with pytest.raises(Exception, match="500"):
            asyncio.run(run(_payload(model.base_url, resources.base_url, tmp_path), tmp_path, Observer()))
    finally:
        model.close()


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
    certified = {t["function"]["name"]: t for t in _CERTIFIED}
    assert [t for t in tools if t["function"]["name"] != "code_exec"] == [
        certified[name] for name in ("finish", "abandon_task_finish", "web_search", "fetch_web_page")
    ]


def test_remote_tool_schema_is_the_rows_schema():
    spec = next(t for t in _TOOLS if t["name"] == "finish")

    tool = remote_tool(spec, client=None)

    assert to_openai_tools({tool.name: tool})[0]["function"]["parameters"] == spec["parameters"]


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
