# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional offline boundary check using the actual pinned CLI and a localhost fake API."""

import json
import os
import socket
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel

from nemo_gym.base_resources_server import BaseResourcesServerConfig, SimpleResourcesServer
from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.mcp_auto_exposure import install_auto_exposure
from nemo_gym.responses_streaming import flatten_namespace_tools, synthesize_responses_sse
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.codex_agent.app import parse_exec_jsonl, toml_dumps
from responses_api_agents.codex_agent.tests.test_sandbox_sessions import close_body, seed
from responses_api_agents.codex_agent.tests.test_sandbox_sessions import setup as setup


def pinned_cli() -> str:
    binary = os.environ.get("CODEX_TEST_BIN")
    if not binary or not Path(binary).is_file():
        pytest.skip("Set CODEX_TEST_BIN to an existing Codex 0.144.4 executable; never auto-install for this test")
    version = subprocess.run([binary, "--version"], capture_output=True, text=True, errors="replace", timeout=10)
    if version.returncode or version.stdout.strip() != "codex-cli 0.144.4":
        pytest.skip("This boundary regression targets Codex 0.144.4")
    return binary


@pytest.mark.parametrize("recover", [False, True])
def test_pinned_cli_incomplete_stream_retries_preserve_usage_gaps(tmp_path: Path, setup, recover: bool) -> None:
    binary = pinned_cli()
    reasoning = "Inspect arithmetic before editing."
    response = {
        "id": "resp_incomplete",
        "object": "response",
        "created_at": 1,
        "model": "gym-policy-model",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [
            {"id": "rs_partial", "type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning}]}
        ],
        "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    }
    requests = []
    completed = {
        **response,
        "id": "resp_complete",
        "status": "completed",
        "incomplete_details": None,
        "output": [
            {
                "id": "msg_done",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Done.", "annotations": []}],
            }
        ],
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            terminal = completed if recover and len(requests) > 1 else response
            wire = "".join(synthesize_responses_sse(terminal)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = tmp_path / "home"
    home.mkdir()
    agent, sandbox = setup
    config = agent._build_config(f"http://127.0.0.1:{server.server_port}/v1")
    (home / "config.toml").write_text(toml_dumps(config))
    try:
        result = subprocess.run(
            [
                binary,
                "exec",
                "--json",
                "--ephemeral",
                "--skip-git-repo-check",
                "--cd",
                str(tmp_path),
                "--",
                "Say done.",
            ],
            env={
                "PATH": os.environ["PATH"],
                "HOME": str(home),
                "CODEX_HOME": str(home),
                "OPENAI_API_KEY": "synthetic",
            },
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert result.returncode == (0 if recover else 1), result.stdout + result.stderr
    assert len(requests) == (2 if recover else 6), "Pinned CLI defaults allow five reconnects, then must terminate"
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert any(event["type"] == "turn.completed" for event in events) == recover
    failed = [event for event in events if event["type"] == "turn.failed"]
    output, usage = parse_exec_jsonl(result.stdout, structured_reasoning=True, include_partial=True)
    if recover:
        assert not failed
        assert [item.type for item in output] == ["reasoning", "message"]
        assert output[0].summary[0].text == reasoning
        assert len({item.id for item in output}) == 2
    else:
        assert len(failed) == 1
        assert "max_output_tokens" in failed[0]["error"]["message"]
        assert output and all(item.type == "reasoning" for item in output)
        assert all(item.summary[0].text == reasoning for item in output)
        assert any("max_output_tokens" in error for error in usage["errors"])

    # Replay actual pinned-CLI events through native response + close. No model call from
    # an incomplete attempt appears in turn.completed, even when a later retry succeeds.
    sandbox.events = "\n".join(json.dumps([float(i), event]) for i, event in enumerate(events))
    sandbox.result["return_code"] = result.returncode
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        activated = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "Say done."})
        assert activated.status_code == 200, activated.text
        native = activated.json()
        assert native["status"] == ("completed" if recover else "incomplete")
        assert native["error"] is None
        if not recover:
            assert native["incomplete_details"]["reason"] == "max_output_tokens"
        assert native["usage"]["total_tokens"] == (10 if recover else 0)
        assert native["usage"]["total_tokens"] < sum(response["usage"]["total_tokens"] for _ in requests)
        # The fake backend omitted both details; CLI zeros cannot prove known zero usage.
        assert native["usage"]["input_tokens_details"]["cached_tokens"] is None
        assert native["usage"]["output_tokens_details"]["reasoning_tokens"] is None
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id)).json()
        assert "partial_model_usage_unavailable" in {gap["code"] for gap in closed["agent_observations"]["gaps"]}


def test_pinned_cli_failed_turn_retains_transcript_usage(tmp_path: Path, setup) -> None:
    binary = pinned_cli()
    agent, sandbox = setup
    requests = []
    expected_usage = {"input_tokens": 41, "output_tokens": 8, "total_tokens": 49}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if len(requests) > 1:
                wire = b'{"error":{"message":"controlled rejection","type":"invalid_request_error"}}'
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
            else:
                response = {
                    "id": "resp_tool",
                    "object": "response",
                    "created_at": 1,
                    "model": "gym-policy-model",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_tool",
                            "call_id": "call_tool",
                            "name": "exec_command",
                            "arguments": json.dumps({"cmd": "printf usage_probe"}),
                            "status": "completed",
                        }
                    ],
                    "usage": expected_usage,
                }
                wire = "".join(synthesize_responses_sse(response)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = tmp_path / "home"
    home.mkdir()
    config = agent._build_config(f"http://127.0.0.1:{server.server_port}/v1")
    (home / "config.toml").write_text(toml_dumps(config))
    try:
        result = subprocess.run(
            [binary, "exec", "--json", "--skip-git-repo-check", "--cd", str(tmp_path), "--", "Run the shell check."],
            env={
                "PATH": os.environ["PATH"],
                "HOME": str(home),
                "CODEX_HOME": str(home),
                "OPENAI_API_KEY": "synthetic",
            },
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert result.returncode == 1, result.stdout + result.stderr
    assert len(requests) == 2
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert any(event["type"] == "turn.failed" for event in events)
    assert not any(event["type"] == "turn.completed" for event in events)
    transcripts = sorted((home / "sessions").rglob("*.jsonl"))
    assert transcripts, "The real CLI must persist its transcript for the usage fallback"
    transcript = "\n".join(path.read_text() for path in transcripts)
    sandbox.events = "\n".join(json.dumps([float(i), event]) for i, event in enumerate(events))
    sandbox.result["return_code"] = result.returncode
    original_publish = sandbox.publish

    def publish():
        original_publish()
        sandbox.files[f"{sandbox.directory}/native_session.jsonl"] = transcript
        sandbox.files[f"{sandbox.directory}/model_attempts.jsonl"] = '{"status":200}\n{"status":400}\n'

    sandbox.publish = publish
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        response = client.post("/ng-rollout/codex-smoke-a2/v1/responses", json={"input": "Run the shell check."})
        response.raise_for_status()
        body = response.json()
        assert body["status"] == "failed"
        assert {key: body["usage"][key] for key in expected_usage} == expected_usage
        assert any(item["type"] == "function_call_output" for item in body["output"])
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        closed.raise_for_status()
        observations = closed.json()["agent_observations"]
        assert observations["records"][0]["status"] == "failed"
        assert observations["records"][0]["error_type"] == "provider_error"
        assert "agent_artifact_record_unparseable" not in {gap["code"] for gap in observations["gaps"]}


class _RecordBody(BaseModel):
    value: str


class _Recorder(SimpleResourcesServer):
    """A resources server with one session-scoped route, exposed over MCP like any Gym tool route."""

    calls: dict[str, list[str]] = {}

    async def verify(self, body):
        pass

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        @app.post("/record")
        async def record(body: _RecordBody, request: Request) -> str:
            """Record a value for this session."""
            self.calls.setdefault(request.session[SESSION_ID_KEY], []).append(body.value)
            return f"recorded {body.value}"

        return app


def test_pinned_cli_calls_a_granted_gym_mcp_tool(tmp_path: Path, setup) -> None:
    """The adapter's config for a session MCP grant lets the pinned CLI call a real Gym MCP route."""
    binary = pinned_cli()
    resources = _Recorder(
        config=BaseResourcesServerConfig(
            host="127.0.0.1", port=0, entrypoint="", name="recorder", expose_tools_over_mcp=True
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    app = resources.setup_webserver()
    install_auto_exposure(resources, app)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    resources_server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    resources_thread = threading.Thread(target=resources_server.run, kwargs={"sockets": [listener]}, daemon=True)

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            # Flatten and restore namespaced tools the way a Gym model server does.
            _, ns_map = flatten_namespace_tools(request.get("tools"))
            if len(requests) == 1:
                # Codex 0.144.4 offers each MCP server as the namespace mcp__<server>.
                (name,) = [name for name, namespace in ns_map.items() if namespace == ("mcp__recorder", "record")]
                item = {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": name,
                    "arguments": json.dumps({"value": "hello"}),
                    "status": "completed",
                }
            else:
                item = {
                    "id": "msg_done",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Done.", "annotations": []}],
                }
            response = {
                "id": f"resp_{len(requests)}",
                "object": "response",
                "created_at": 1,
                "model": "gym-policy-model",
                "status": "completed",
                "output": [item],
                "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            }
            wire = "".join(synthesize_responses_sse(response, ns_map)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

        def log_message(self, format: str, *args: object) -> None:
            pass

    model_server = HTTPServer(("127.0.0.1", 0), Handler)
    model_thread = threading.Thread(target=model_server.serve_forever, daemon=True)
    home = tmp_path / "home"
    home.mkdir()
    resources_thread.start()
    model_thread.start()
    try:
        deadline = time.monotonic() + 10
        while not resources_server.started:
            assert time.monotonic() < deadline, "resources server did not start"
            time.sleep(0.05)
        base_url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        seed_request = urllib.request.Request(
            f"{base_url}/seed_session", data=b"{}", headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(seed_request, timeout=10) as seeded:
            metadata = json.loads(seeded.read())["mcp"]
        grant = {
            "kind": "mcp",
            "name": metadata["server_name"],
            "required": True,
            "connection": {"url": f"{base_url}{metadata['url_path']}", "headers": metadata["headers"]},
        }
        agent, _ = setup
        request = AgentSeedSessionRequest(**(seed().model_dump(mode="json") | {"tool_accesses": [grant]}))
        config = agent._build_config(
            f"http://127.0.0.1:{model_server.server_port}/v1", mcp_servers=agent._granted_mcp_servers(request)[0]
        )
        (home / "config.toml").write_text(toml_dumps(config))
        result = subprocess.run(
            [binary, "exec", "--json", "--ephemeral", "--skip-git-repo-check", "--cd", str(tmp_path), "--", "Record."],
            env={"PATH": os.environ["PATH"], "HOME": str(home), "CODEX_HOME": str(home), "OPENAI_API_KEY": "x"},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
        )
    finally:
        model_server.shutdown()
        model_server.server_close()
        model_thread.join(timeout=2)
        resources_server.should_exit = True
        resources_thread.join(timeout=5)
    assert result.returncode == 0, result.stdout + result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines()]
    calls = [event["item"] for event in events if (event.get("item") or {}).get("type") == "mcp_tool_call"]
    assert calls and calls[-1]["status"] == "completed", events
    assert (calls[-1]["server"], calls[-1]["tool"]) == ("recorder", "record")
    # One call reached the route, under the one session the grant's token names.
    assert list(resources.calls.values()) == [["hello"]]
    outputs = [item for item in requests[1]["input"] if item.get("type") == "function_call_output"]
    assert outputs and "recorded hello" in json.dumps(outputs)


@pytest.mark.parametrize("required", [True, False])
def test_pinned_cli_with_an_unreachable_mcp_server(tmp_path: Path, setup, required: bool) -> None:
    """A required grant must fail the activation instead of letting the model run without its tools.

    An optional grant that does not connect is left out, and the run reaches the model.
    """
    binary = pinned_cli()
    unused = socket.socket()
    unused.bind(("127.0.0.1", 0))
    port = unused.getsockname()[1]
    unused.close()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requests.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(500)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    model_server = HTTPServer(("127.0.0.1", 0), Handler)
    model_thread = threading.Thread(target=model_server.serve_forever, daemon=True)
    model_thread.start()
    home = tmp_path / "home"
    home.mkdir()
    grant = {
        "kind": "mcp",
        "name": "recorder",
        "required": required,
        "connection": {"url": f"http://127.0.0.1:{port}/mcp", "headers": {}},
    }
    agent, _ = setup
    request = AgentSeedSessionRequest(**(seed().model_dump(mode="json") | {"tool_accesses": [grant]}))
    config = agent._build_config(
        f"http://127.0.0.1:{model_server.server_port}/v1", mcp_servers=agent._granted_mcp_servers(request)[0]
    )
    # The model server always fails; do not wait for Codex's retries.
    config["model_providers"]["gym"].update(request_max_retries=0, stream_max_retries=0)
    (home / "config.toml").write_text(toml_dumps(config))
    try:
        result = subprocess.run(
            [binary, "exec", "--json", "--ephemeral", "--skip-git-repo-check", "--cd", str(tmp_path), "--", "Record."],
            env={"PATH": os.environ["PATH"], "HOME": str(home), "CODEX_HOME": str(home), "OPENAI_API_KEY": "x"},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
        )
    finally:
        model_server.shutdown()
        model_server.server_close()
        model_thread.join(timeout=2)
    started = any(json.loads(line).get("type") == "turn.started" for line in result.stdout.splitlines())
    if not required:
        assert started and requests, result.stdout + result.stderr
        return
    assert result.returncode != 0, result.stdout + result.stderr
    # The adapter's stderr-tail error path relies on both: no turn starts, and the reason is on stderr.
    assert not started
    assert "required MCP servers failed to initialize" in result.stderr
    assert not requests, "the model must not be called without the required tools"
