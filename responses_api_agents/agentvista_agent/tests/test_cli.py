# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real pinned CLI -> MCP -> Python/image roundtrip with a local model fixture."""

import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest


if not os.environ.get("AGENTVISTA_ROOT"):
    pytest.skip("Set AGENTVISTA_ROOT to the pinned upstream source", allow_module_level=True)
sys.path.insert(0, os.environ["AGENTVISTA_ROOT"])

from PIL import Image

from responses_api_agents.agentvista_agent.runtime.data import inventory


@pytest.mark.skipif(
    not os.environ.get("AGENTVISTA_TEST_OPENCODE"),
    reason="Set AGENTVISTA_TEST_OPENCODE for pinned CLI smoke",
)
@pytest.mark.parametrize("budget_exhausted,judge_error", [(False, False), (True, False), (True, True)])
def test_real_cli_image_and_state_roundtrip(tmp_path, monkeypatch, budget_exhausted, judge_error):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body["model"] == "fixture-judge":
                message, finish = (
                    {
                        "role": "assistant",
                        "content": "invalid judgment"
                        if judge_error
                        else f"Score: {int(not budget_exhausted)}\nExplanation: Fixture judgment.",
                    },
                    "stop",
                )
            else:
                requests.append(body)
                index = len(requests)
                if budget_exhausted:
                    message = {"role": "assistant", "content": None, "reasoning_content": "Still thinking."}
                    finish = "length"
                elif index <= 2:
                    code = (
                        "value = 41\noriginal_image.crop((0, 0, 100, 100))"
                        if index == 1
                        else "print(value + 1)\nprint(tool_image_1.size)"
                    )
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"action-{index}",
                                "type": "function",
                                "function": {
                                    "name": "agentvista_code_interpreter",
                                    "arguments": json.dumps({"code": code}),
                                },
                            }
                        ],
                    }
                    finish = "tool_calls"
                else:
                    message, finish = (
                        {"role": "assistant", "content": "<answer>42</answer>"},
                        "stop",
                    )
            base = {"id": "fixture-" + str(len(requests)), "created": 1, "model": body["model"]}
            usage = {"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40}
            if body.get("stream"):
                delta = dict(message)
                if "tool_calls" in delta:
                    delta["tool_calls"] = [{"index": 0, **delta["tool_calls"][0]}]
                chunk = {
                    **base,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                end = {
                    **base,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    "usage": usage,
                }
                payload = (
                    "data: " + json.dumps(chunk) + "\n\ndata: " + json.dumps(end) + "\n\ndata: [DONE]\n\n"
                ).encode()
                content_type = "text/event-stream"
            else:
                payload = json.dumps(
                    {
                        **base,
                        "object": "chat.completion",
                        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                        "usage": usage,
                    }
                ).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    for key, value in {
        "SERPAPI_KEY": "fixture",
        "AGENTVISTA_VISIT_API_KEY": "fixture",
        "AGENTVISTA_VISIT_API_ENDPOINT": endpoint + "/chat/completions",
        "AGENTVISTA_VISIT_MODEL_NAME": "fixture-helper",
        "VERIFIER_API_KEY": "fixture",
        "VERIFIER_END_POINT": endpoint,
        "VERIFIER_MODEL_NAME": "fixture-judge",
    }.items():
        monkeypatch.setenv(key, value)
    image = tmp_path / "input.png"
    Image.new("RGB", (320, 200), "red").save(image)
    dataset = tmp_path / "tasks.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": "fixture",
                    "problem": "Use Python and the image; report 42.",
                    "images": [image.name],
                    "solution": "42",
                }
            ]
        )
    )
    task = inventory(dataset, tmp_path)["fixture"]
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    mounted = tmp_path / "mounted"
    mounted.symlink_to(persistent, target_is_directory=True)
    try:
        config = {
            "output": str(mounted / "episode"),
            "dataset": str(dataset),
            "image_folder": str(tmp_path),
            "task_id": "fixture",
            "source_sha256": task["source_sha256"],
            "command": os.environ["AGENTVISTA_TEST_OPENCODE"],
            "endpoint": endpoint,
            "model": "fixture",
            "context_tokens": 65536,
            "output_tokens": 32768,
            "max_steps": 6,
            "agent_timeout": 120,
        }
        config_path = tmp_path / "worker.json"
        config_path.write_text(json.dumps(config))
        subprocess.run(
            [sys.executable, "-m", "responses_api_agents.agentvista_agent.runtime.worker", str(config_path)],
            check=True,
            timeout=180,
        )
        result = json.loads((mounted / "episode/result.json").read_text())
        assert result["mask_sample"] is judge_error, result
        assert result["reward"] == int(not budget_exhausted)
        assert result["agent_output"] == ("" if budget_exhausted else "<answer>42</answer>")
        assert result["artifacts"] == str(mounted / "episode")
        assert (persistent / "episode/events.jsonl").is_file()
        assert len(requests) == (1 if budget_exhausted else 3)
        assert all(row["max_tokens"] == 32768 for row in requests)
        assert (persistent / "episode/tools/tools_listed").is_file()
        expected = {"agentvista_" + name for name in ("web_search", "image_search", "visit", "code_interpreter")}
        assert {tool["function"]["name"] for tool in requests[0]["tools"]} == expected

        def images(request):
            return [
                part
                for message in request["messages"]
                for part in message.get("content", [])
                if isinstance(part, dict) and part.get("type") == "image_url"
            ]

        assert len(images(requests[0])) == 1
        native = [e for e in result["trajectory"] if e.get("type") == "tool_use"]
        if budget_exhausted:
            assert not native
            assert result["execution_health"]["termination"] == "output_budget_exhausted"
            assert result["execution_health"]["finish_reason"] == "length"
            if judge_error:
                assert result["failure_kind"] == "agentvista:verifier"
                assert result["verify"]["status"] == "ERROR"
            else:
                assert result["verify"]["status"] == "FAIL"
        else:
            assert len(images(requests[1])) == 2
            assert "42" in json.dumps(requests[2]["messages"])
            assert len(native) == 2 and all(e["part"]["state"]["status"] == "completed" for e in native)
            assert Path(result["artifacts"], "tools/tools.jsonl").is_file()
    finally:
        server.shutdown()
        server.server_close()
