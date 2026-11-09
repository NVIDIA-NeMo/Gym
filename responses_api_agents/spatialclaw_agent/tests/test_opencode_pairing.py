# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in controlled smoke: real OpenCode, authenticated tools, and a real Python kernel.

This test uses a scripted local model. It is not model accuracy, RL, or P0 evidence.
"""

import asyncio
import base64
import hashlib
import io
import json
import os
import subprocess
import threading
from concurrent.futures import Future
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from nemo_gym.agents import AgentSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime.mcp_auto_exposure import _to_result
from resources_servers.spatialclaw.session_runtime import SpatialClawSessionRuntime, context_result
from responses_api_agents.opencode_agent.app import OpenCodeAgent, OpenCodeAgentConfig


PINNED_VERSION = "1.18.34"
PINNED_SHA256 = "9ca0b9953d49997601655e54f846a3efa464f237e47c6f1b04716d0f2e64c4c2"


@contextmanager
def http_server(handler):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            status, kind, payload = handler(body, self.headers)
            self.send_response(status)
            self.send_header("content-type", kind)
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def completion(body, *, name=None, arguments=None):
    def chunk(delta, finish=None, choices=None, usage=None):
        item = {
            "id": "scripted-call",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": body.get("model", "synthetic"),
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}] if choices is None else choices,
        }
        if usage:
            item["usage"] = usage
        return f"data: {json.dumps(item)}\n\n"

    if name:
        call = {
            "index": 0,
            "id": f"call-{name}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
        stream = chunk({"role": "assistant", "tool_calls": [call]}) + chunk({}, "tool_calls")
    else:
        stream = chunk({"role": "assistant", "content": "Done."}) + chunk({}, "stop")
    stream += chunk(None, choices=[], usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    return 200, "text/event-stream", (stream + "data: [DONE]\n\n").encode()


@pytest.mark.skipif(not os.environ.get("SPATIALCLAW_OPENCODE_BIN"), reason="requires pinned OpenCode binary")
@pytest.mark.skipif(not os.environ.get("SPATIALCLAW_ROOT"), reason="requires pinned SpatialClaw source")
def test_opencode_uses_shared_context_image_and_kernel(tmp_path, monkeypatch):
    binary = Path(os.environ["SPATIALCLAW_OPENCODE_BIN"]).resolve()
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == PINNED_SHA256
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=30)
    assert version.returncode == 0 and version.stdout.strip() == PINNED_VERSION
    monkeypatch.syspath_prepend(os.environ["SPATIALCLAW_ROOT"])
    from spatial_agent.kernel.manager import JupyterKernelManager

    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(key, raising=False)
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    def run(awaitable):
        future: Future = asyncio.run_coroutine_threadsafe(awaitable, loop)
        return future.result(timeout=60)

    kernel = JupyterKernelManager(timeout_sec=30)
    runtime = SpatialClawSessionRuntime.new("live-pair", "one", str(tmp_path / "resources"))
    runtime.kernel = kernel
    runtime.spatial_config = SimpleNamespace(max_steps=3, timeout_sec=30)
    image_stream = io.BytesIO()
    Image.new("RGB", (4, 4), "green").save(image_stream, format="PNG")
    image_base64 = base64.b64encode(image_stream.getvalue()).decode()
    runtime.context = {
        "instruction": "What is the letter?",
        "images": [],
        "metadata": {},
        "session_id": "live-pair",
        "system_prompt": "Use execute to submit ReturnAnswer('A').",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect this image, then execute ReturnAnswer('A')."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
    }
    requests, tool_calls, authenticated = [], [], []

    def mcp(body, headers):
        if headers.get("x-smoke-token") != "episode-only":
            return 401, "application/json", b"{}"
        authenticated.append(body["method"])
        if body["method"] == "notifications/initialized":
            return 202, "application/json", b""
        if body["method"] == "initialize":
            result = {
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "spatialclaw-smoke", "version": "1"},
            }
        elif body["method"] == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "get_context",
                        "description": "Get native task context and images",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                    {
                        "name": "execute",
                        "description": "Execute a Python cell",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"code": {"type": "string"}},
                            "required": ["code"],
                        },
                    },
                ]
            }
        elif body["method"] == "tools/call":
            name = body["params"]["name"]
            tool_calls.append(name)
            if name == "get_context":
                payload = context_result(runtime.context)
            else:
                assert name == "execute"
                payload = run(runtime.execute(body["params"]["arguments"]["code"], 30))
            result = _to_result(payload).model_dump(mode="json", by_alias=True, exclude_none=True)
        else:
            raise AssertionError(f"Unexpected MCP method: {body['method']}")
        return 200, "application/json", json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result}).encode()

    def model(body, headers):
        functions = [item.get("function", item) for item in body.get("tools", [])]
        names = {item.get("name", "") for item in functions}
        if not any(name.endswith("_get_context") for name in names):
            return completion(body)  # title/summary helpers do not drive the task.
        requests.append(body)
        if len(requests) == 1:
            return completion(body, name=next(n for n in names if n.endswith("_get_context")), arguments={})
        if len(requests) == 2:
            assert image_base64 in json.dumps(body), "MCP image did not reach the real OpenCode model call"
            return completion(
                body, name=next(n for n in names if n.endswith("_execute")), arguments={"code": "ReturnAnswer('A')"}
            )
        return completion(body)

    try:
        run(kernel.start())
        initialized = run(
            kernel.execute(
                "import builtins\ndef ReturnAnswer(value):\n    builtins._return_answer_result = {'text': str(value)}\n"
            )
        )
        assert initialized.error is None
        with http_server(model) as model_url, http_server(mcp) as mcp_url:
            client = MagicMock(spec=ServerClient)
            client.global_config_dict = {
                "policy": {"responses_api_models": {"openai_model": {"host": "127.0.0.1", "port": 9999}}}
            }
            client._build_server_base_url.return_value = model_url
            config = OpenCodeAgentConfig(
                host="127.0.0.1",
                port=8001,
                name="opencode",
                entrypoint="app.py",
                num_workers=1,
                model_server={"type": "responses_api_models", "name": "policy"},
                opencode_version=PINNED_VERSION,
                sandbox_provider="local-smoke",
                sandbox_config={"workdir": str(tmp_path / "task")},
                sandbox_executable_path=str(binary),
                sandbox_executable_sha256=PINNED_SHA256,
                timeout=120,
                sandbox_install_timeout_seconds=120,
                session_close_timeout_seconds=30,
                context_window=32000,
                max_output_tokens=4096,
                mcp_tool_timeout_seconds=60,
                opencode_config={"permission": {"*": "deny", "spatialclaw_*": "allow"}, "compaction": {"auto": False}},
            )
            agent = OpenCodeAgent(config=config, server_client=client)
            seed = AgentSeedSessionRequest(
                agent_session_id="live-pair",
                episode_id=EpisodeId(rollout_id="live-pair"),
                task_id=TaskId(taskset="spatialclaw:smoke", task_id="one"),
                tool_accesses=[
                    {
                        "kind": "mcp",
                        "name": "spatialclaw",
                        "required": True,
                        "connection": {
                            "transport": "streamable_http",
                            "url": mcp_url + "/mcp",
                            "headers": {"x-smoke-token": "episode-only"},
                        },
                    }
                ],
            )
            with (
                patch(
                    "responses_api_agents.opencode_agent.app.get_global_config_dict",
                    return_value={"local-smoke": {"local": {"workspace_root": str(tmp_path / "sandboxes")}}},
                ),
                TestClient(agent.setup_webserver()) as http,
            ):
                seeded = http.post("/v1/agent_sessions", json=seed.model_dump(mode="json"))
                assert seeded.status_code == 200, seeded.text
                response = http.post(
                    "/ng-rollout/live-pair/v1/responses",
                    json={"input": "Use get_context, inspect the image, then execute ReturnAnswer('A')."},
                )
                assert response.status_code == 200, response.text
                assert response.json()["status"] == "completed", response.text
                closed = http.post(
                    "/v1/agent_sessions/close",
                    json={"agent_session_id": seed.agent_session_id, "episode_id": seed.episode_id.model_dump()},
                )
                assert closed.status_code == 200, closed.text
            assert requests and len(requests) == 3
            assert tool_calls == ["get_context", "execute"]
            assert runtime.accepted_answer == "A"
            assert "tools/call" in authenticated
            assert not any(record.state is not None for record in agent._session_records.values())
            (tmp_path / "model-requests.json").write_text(json.dumps(requests, indent=2))
    finally:
        run(kernel.shutdown())
        assert not kernel.is_running
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5)
        loop.close()
