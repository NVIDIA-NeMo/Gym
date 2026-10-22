# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real pinned SDK and shell tool against a deterministic model endpoint."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
from threading import Thread
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from responses_api_agents.deepseek_harness_agent import runner
from responses_api_agents.deepseek_harness_agent.trajectory import convert_events
from responses_api_models.deepseek_model.app import DeepSeekModel
from responses_api_models.openai_model.app import SimpleModelServerConfig


try:
    sdk_available = version("deepseek-harness-sdk") == runner.DSH_VERSION
except PackageNotFoundError:
    sdk_available = False


@pytest.mark.skipif(not sdk_available, reason=f"Requires deepseek-harness-sdk=={runner.DSH_VERSION}")
@pytest.mark.parametrize("profile", ["sdk-minimal", "sdk"])
@pytest.mark.parametrize("reasoning_effort", ["high", "max"])
def test_real_sdk_executes_tool_and_exports_gym_trajectory(tmp_path, profile, reasoning_effort, monkeypatch):
    requests = []

    async def complete(**body):
        requests.append(body)
        if len(requests) == 1:
            tool = next(t["function"] for t in body["tools"] if t["function"]["name"] == "bash")
            arguments = {"command": "printf 'sdk-ok' > proof.txt; cat proof.txt"}
            if "description" in tool["parameters"]["properties"]:
                arguments["description"] = "Write and read the smoke-test artifact"
            message = {
                "role": "assistant",
                "content": None,
                "reasoning_content": "I will run the command.",
                "tool_calls": [
                    {
                        "id": "call_proof",
                        "type": "function",
                        "function": {"name": tool["name"], "arguments": json.dumps(arguments)},
                    }
                ],
            }
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": "sdk-ok"}
            finish = "stop"
        return {
            "id": f"chat_{len(requests)}",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
        }

    model = DeepSeekModel(
        config=SimpleModelServerConfig(
            host="127.0.0.1",
            port=8081,
            name="policy_model",
            entrypoint="app.py",
            openai_base_url="https://example.invalid/v1",
            openai_api_key="test-only",
            openai_model="deepseek-v4-flash",
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    model._client = MagicMock(create_chat_completion=AsyncMock(side_effect=complete))

    class Model(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            response = gym.post(self.path, json=body)
            self.send_response(response.status_code)
            self.send_header("Content-Type", response.headers["Content-Type"])
            self.send_header("Content-Length", str(len(response.content)))
            self.end_headers()
            self.wfile.write(response.content)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with TestClient(model.setup_webserver()) as gym, ThreadingHTTPServer(("127.0.0.1", 0), Model) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        config = {
            "prompt": "Write sdk-ok to proof.txt and read it back.",
            "session_id": "sdk-smoke",
            "harness": {
                "model": "deepseek-v4-flash",
                "profile": profile,
                "reasoning_effort": reasoning_effort,
                "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                "api_key": "test-only",
                "env": {"DSH_PERMISSION_MODE": "danger-full-access"},
                "initialize_timeout_seconds": 30,
                "request_timeout_seconds": 20,
            },
        }
        config_path = run_dir / "input.json"
        config_path.write_text(json.dumps(config))
        try:
            monkeypatch.chdir(workspace)
            exit_code = runner.run(config_path)
        finally:
            server.shutdown()
            thread.join(timeout=5)
    summary = json.loads((run_dir / "result.json").read_text())
    assert exit_code == 0, summary
    assert summary["finish_reason"] == "completed"
    assert (workspace / "proof.txt").read_text() == "sdk-ok"
    assert len(requests) == 2
    assert all(r["reasoning_effort"] == reasoning_effort for r in requests)
    assert all(r["thinking"] == {"type": "enabled"} for r in requests)
    assert any(m["role"] == "tool" and "sdk-ok" in str(m["content"]) for m in requests[1]["messages"])
    assert any(m.get("reasoning_content") == "I will run the command." for m in requests[1]["messages"])
    notifications = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    output, usage = convert_events(notifications, "sdk-smoke")
    assert [item.type for item in output] == ["reasoning", "function_call", "function_call_output", "message"]
    assert output[1].call_id == output[2].call_id == "call_proof"
    assert output[-1].content[0].text == "sdk-ok"
    assert usage.total_tokens == 46
    assert not (workspace / ".dsh").exists()
