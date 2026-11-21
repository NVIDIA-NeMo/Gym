# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A scripted stand-in for the policy model: it plays by writing ``python`` tool calls that act.

``stub_reply`` builds a vLLM-style chat completion (``reasoning_content`` + one ``python`` tool call) for a request.
``python stub_model.py --port N --log FILE`` serves it as an OpenAI-compatible ``/v1/chat/completions`` endpoint
(what a vLLM server with a reasoning parser returns), logging every request body to FILE for test assertions.
"""

import argparse
import itertools
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


KEY_ACTIONS = ("UP", "LEFT", "DOWN", "RIGHT", "SPACE")
_counter = itertools.count()
_lock = threading.Lock()


def stub_code(step: int) -> str:
    """Python the harness runs: a keyboard action on even steps, a click on odd steps (games accept one or both)."""
    if step % 2 == 0:
        return f"print(action([{KEY_ACTIONS[(step // 2) % len(KEY_ACTIONS)]!r}]))\n"
    row, col = 8 + (step * 7) % 48, 8 + (step * 13) % 48
    return f"print(action([{{'action': 'MOUSE', 'row': {row}, 'col': {col}}}]))\n"


def stub_reply(request: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        step = next(_counter)
    tools = request.get("tools") or []
    prompt_chars = sum(len(json.dumps(m.get("content"))) for m in request.get("messages") or [])
    message: dict[str, Any] = {
        "role": "assistant",
        "reasoning_content": f"Stub reasoning for step {step}: try one action and look at the result.",
        "content": "",
    }
    finish = "stop"
    if tools and not request.get("continue_final_message"):
        message["tool_calls"] = [
            {
                "id": f"call_stub_{step}",
                "type": "function",
                "function": {"name": tools[0]["function"]["name"], "arguments": json.dumps({"code": stub_code(step)})},
            }
        ]
        finish = "tool_calls"
    else:
        message["content"] = "Stub summary: no findings."
    return {
        "id": f"chatcmpl-stub-{step}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.get("model") or "stub",
        "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
        "usage": {
            "prompt_tokens": max(1, prompt_chars // 4),
            "completion_tokens": 40,
            "total_tokens": prompt_chars // 4 + 40,
        },
    }


def serve(port: int, log_path: str) -> None:
    log_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._send(200, {"object": "list", "data": [{"id": "stub", "object": "model"}]})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with log_lock, open(log_path, "a") as log:
                log.write(json.dumps({"path": self.path, "body": body}) + "\n")
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self._send(404, {"error": f"no route {self.path}"})
                return
            self._send(200, stub_reply(body))

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    serve(args.port, args.log)
