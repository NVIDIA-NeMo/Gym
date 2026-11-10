# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sandbox runner, run as a process over the layout a tool sandbox gets, and driven over HTTP."""

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from importlib import metadata
from pathlib import Path

import pytest

from resources_servers.partner_if import example_tools
from resources_servers.partner_if.task_data import ToolBackendBundle
from resources_servers.partner_if.tool_runner import TOKEN_HEADER
from resources_servers.partner_if.tool_sandbox import sandbox_files


PYDANTIC_PIN = f"pydantic=={metadata.version('pydantic')}"
NOTES = {
    "entrypoint": "example_tools:NotesBackend",
    "files": {"example_tools.py": Path(example_tools.__file__).read_text(), "notes.json": '{"greeting": "HELLO"}'},
    "fixture": "notes.json",
    "requirements": [PYDANTIC_PIN],
}
# A backend that needs nothing beyond the standard library, and fails on request.
PLAIN = """
from resources_servers.partner_if.tool_backend import ToolCallError


class Plain:
    requires_final_answer = False

    def __init__(self, fixture_path):
        self.greeting = fixture_path.read_text()

    def tools(self, row):
        return {name: {"description": name} for name in ("greet", "fail", "unjson")}

    def new_state(self, row):
        return []

    def call(self, state, name, arguments):
        if name == "fail":
            raise RuntimeError("backend bug")
        if name == "unjson":
            return {"not", "json"}
        if not arguments:
            raise ToolCallError("Say who to greet")
        state.append(arguments["who"])
        return f"{self.greeting}, {arguments['who']}"

    def artifacts(self, state):
        return {"greeted": ",".join(state)}

    def receipts(self, state):
        return {"greeted": state}
"""
PLAIN_BUNDLE = {
    "entrypoint": "plain:Plain",
    "files": {"plain.py": PLAIN, "hello.txt": "Hello"},
    "fixture": "hello.txt",
}


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def runner(tmp_path):
    """Lay out a sandbox as tool_sandbox.py does and start its runner; returns a function to make requests."""
    processes = []

    def start(bundle, *, token="token", python_flags=("-I",)):
        bundle = ToolBackendBundle.model_validate(bundle)
        root = tmp_path / f"sandbox-{len(processes)}"
        port = free_port()
        config = {
            "entrypoint": bundle.entrypoint,
            "fixture": bundle.fixture,
            "requirements": bundle.requirements,
            "host": "127.0.0.1",
            "port": port,
            "token": token,
        }
        for path, text in sandbox_files(bundle, config, install=False).items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(text)
        command = [sys.executable, *python_flags, str(root / "tool_runner.py"), str(root / "runner.json")]
        processes.append(subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

        def send(method, route, body=None, *, token=token):
            data = None if body is None else json.dumps(body).encode()
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/{route}", data=data, method=method, headers={TOKEN_HEADER: token}
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    return response.status, json.loads(response.read())
            except urllib.error.HTTPError as error:
                return error.code, json.loads(error.read())

        deadline = time.monotonic() + 30
        while True:
            try:
                status, payload = send("GET", "requires_final_answer")
                if payload.get("status") != "loading":
                    return send
            except (urllib.error.URLError, ConnectionError):
                pass
            assert time.monotonic() < deadline, "The runner did not start"
            time.sleep(0.05)

    yield start
    for process in processes:
        process.terminate()
        process.wait()


ROW = {"row": {"responses_create_params": {"input": [{"role": "user", "content": "Hi"}]}}}


def test_the_runner_serves_a_backends_protocol_for_one_episode(runner):
    send = runner(NOTES)
    assert send("GET", "requires_final_answer") == (200, {"status": "ready", "requires_final_answer": True})
    status, tools = send("POST", "tools", ROW)
    assert status == 200 and set(tools["tools"]) == {"lookup", "append"}
    assert send("POST", "call", {"name": "lookup", "arguments": {"key": "greeting"}})[0] == 409
    assert send("POST", "new_state", ROW) == (200, {})
    assert send("POST", "new_state", ROW)[0] == 409  # One runner holds one episode.
    assert send("POST", "call", {"name": "lookup", "arguments": {"key": "greeting"}}) == (
        200,
        {"output": {"note": "HELLO"}},
    )
    # A ToolCallError is the model's mistake: its message is the output the model sees.
    assert send("POST", "call", {"name": "lookup", "arguments": {"key": "missing"}}) == (
        200,
        {"tool_call_error": "No note has the key 'missing'"},
    )
    assert send("POST", "call", {"name": "append", "arguments": {"text": "HELLO"}}) == (200, {"output": "Appended"})
    assert send("GET", "artifacts") == (200, {"artifacts": {"log": "HELLO"}})
    assert send("GET", "receipts") == (200, {"receipts": {"appended": ["HELLO"]}})
    assert send("POST", "receipts")[0] == 404


def test_only_requests_with_the_sandboxs_token_are_served(runner):
    send = runner(NOTES)
    for token in ("", "guess"):
        assert send("POST", "new_state", ROW, token=token)[0] == 403
    assert send("POST", "new_state", ROW) == (200, {})


def test_the_runner_and_the_protocol_need_only_the_standard_library(runner):
    # -S leaves out site-packages, as in an image with nothing but Python.
    send = runner(PLAIN_BUNDLE, python_flags=("-I", "-S"))
    assert send("GET", "requires_final_answer")[1]["requires_final_answer"] is False
    send("POST", "new_state", ROW)
    assert send("POST", "call", {"name": "greet", "arguments": {"who": "Ada"}}) == (200, {"output": "Hello, Ada"})
    assert send("POST", "call", {"name": "greet", "arguments": {}}) == (200, {"tool_call_error": "Say who to greet"})
    assert send("GET", "receipts") == (200, {"receipts": {"greeted": ["Ada"]}})


def test_other_backend_failures_are_500s_whose_traceback_is_for_the_servers_log(runner):
    send = runner(PLAIN_BUNDLE)
    send("POST", "new_state", ROW)
    status, failure = send("POST", "call", {"name": "fail", "arguments": {}})
    assert status == 500 and "RuntimeError: backend bug" in failure["error"] and "Traceback" in failure["error"]
    status, failure = send("POST", "call", {"name": "unjson", "arguments": {}})
    assert status == 500 and "not JSON serializable" in failure["error"]
    assert send("POST", "call", {"name": "greet", "arguments": {"who": "Ada"}})[0] == 200


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"entrypoint": "example_tools:Missing"}, "AttributeError"),
        ({"files": NOTES["files"] | {"notes.json": "{not json"}}, "JSONDecodeError"),
        (
            {"requirements": ["pydantic==0.0.1"]},
            f"needs pydantic==0.0.1, but pydantic is {metadata.version('pydantic')}",
        ),
        ({"requirements": [PYDANTIC_PIN, "surely-not-installed==1.0"]}, "surely-not-installed is not installed"),
    ],
)
def test_a_backend_that_cannot_load_says_why_on_every_route(runner, change, reason):
    send = runner(NOTES | change)
    for method, route in (("GET", "requires_final_answer"), ("POST", "new_state")):
        status, failure = send(method, route, ROW if method == "POST" else None)
        assert status == 500 and failure["status"] == "failed" and reason in failure["error"]
