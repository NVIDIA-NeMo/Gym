# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real pinned SDK and shell tool against a deterministic model endpoint."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
from threading import Thread

import pytest

from nemo_gym.base_responses_api_model import _validate_chat_params
from nemo_gym.chat_streaming import sanitize_streaming_chat_body
from responses_api_agents.deepseek_harness_agent import runner
from responses_api_agents.deepseek_harness_agent.trajectory import convert_events


try:
    sdk_available = version("deepseek-harness-sdk") == runner.DSH_VERSION
except PackageNotFoundError:
    sdk_available = False


@pytest.mark.skipif(not sdk_available, reason=f"Requires deepseek-harness-sdk=={runner.DSH_VERSION}")
@pytest.mark.parametrize("profile", ["sdk-minimal", "sdk"])
def test_real_sdk_executes_tool_and_exports_gym_trajectory(tmp_path, profile, monkeypatch):
    requests = []

    class Model(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            cleaned, include_usage = sanitize_streaming_chat_body(body)
            assert include_usage
            body = _validate_chat_params(cleaned).model_dump(exclude_unset=True)
            requests.append(body)
            if len(requests) == 1:
                tool = next(t["function"] for t in body["tools"] if t["function"]["name"] == "bash")
                arguments = {"command": "printf 'sdk-ok' > proof.txt; cat proof.txt"}
                if "description" in tool["parameters"]["properties"]:
                    arguments["description"] = "Write and read the smoke-test artifact"
                delta = {
                    "role": "assistant",
                    "reasoning_content": "I will run the command.",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_proof",
                            "type": "function",
                            "function": {
                                "name": tool["name"],
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
                finish = "tool_calls"
            else:
                delta = {"role": "assistant", "content": "sdk-ok"}
                finish = "stop"
            chunk = {
                "id": f"chat_{len(requests)}",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            terminal = chunk | {
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
            }
            payload = (f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(terminal)}\n\ndata: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with ThreadingHTTPServer(("127.0.0.1", 0), Model) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        config = {
            "prompt": "Write sdk-ok to proof.txt and read it back.",
            "session_id": "sdk-smoke",
            "harness": {
                "model": "deepseek-v4-flash",
                "profile": profile,
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
    assert any(m["role"] == "tool" and "sdk-ok" in str(m["content"]) for m in requests[1]["messages"])
    assert any(m.get("reasoning_content") == "I will run the command." for m in requests[1]["messages"])
    notifications = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    output, usage = convert_events(notifications, "sdk-smoke")
    assert [item.type for item in output] == ["reasoning", "function_call", "function_call_output", "message"]
    assert output[1].call_id == output[2].call_id == "call_proof"
    assert output[-1].content[0].text == "sdk-ok"
    assert usage.total_tokens == 46
    assert not (workspace / ".dsh").exists()
