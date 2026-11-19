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
"""A scripted stand-in for vLLM behind the real Gym vLLM model server, for the official harness's tests.

Every reply has a unique content ([stub turn N] ACTION..., which upstream's free-text parser reads) and a unique
reasoning text, so a test can check that each later request resends the reasoning of every earlier reply. Requests
for upstream's compaction summary get a summary. Every request body that reaches "vLLM" is logged.
"""

import argparse
import itertools
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


SUMMARY_SYSTEM_PREFIX = "You create concise conversation summaries"
ACTION_LINE = re.compile(r"^- (RESET|ACTION\d)\b", re.M)


class StubVLLM:
    def __init__(self) -> None:
        self._counter = itertools.count()
        self._lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []
        # reply content -> its reasoning, for every reply sent
        self.reasoning_of: dict[str, str] = {}

    def reply(self, body: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            step = next(self._counter)
            self.requests.append(json.loads(json.dumps(body)))
        messages = body.get("messages") or []
        system = messages[0].get("content") if messages and messages[0].get("role") == "system" else ""
        if isinstance(system, str) and system.startswith(SUMMARY_SYSTEM_PREFIX):
            content = f"[stub summary {step}] Tried several actions; keep exploring."
        else:
            last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
            text = last_user.get("content") if isinstance(last_user.get("content"), str) else ""
            actions = [a for a in ACTION_LINE.findall(text.split("Available actions:")[-1]) if a != "RESET"]
            action = actions[step % len(actions)] if actions else "ACTION1"
            if action == "ACTION6":
                action = f"ACTION6 {8 + (step * 7) % 48} {8 + (step * 13) % 48}"
            content = f"[stub turn {step}] {action}"
        reasoning = f"stub reasoning {step}: look at the board, then act."
        with self._lock:
            self.reasoning_of[content] = reasoning
        prompt_chars = sum(len(json.dumps(m.get("content"))) for m in messages)
        return {
            "id": f"chatcmpl-official-stub-{step}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model") or "stub",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content, "reasoning_content": reasoning},
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": max(1, prompt_chars // 4),
                "completion_tokens": 20,
                "total_tokens": prompt_chars // 4 + 20,
            },
        }


def serve(port: int, log_path: str) -> None:
    stub = StubVLLM()
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
            self._send(200, {"object": "list", "data": [{"id": "stub", "object": "model", "max_model_len": 262144}]})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self._send(404, {"error": f"no route {self.path}"})
                return
            reply = stub.reply(body)
            with log_lock, open(log_path, "a") as log:
                log.write(json.dumps({"body": body, "reply": reply}) + "\n")
            self._send(200, reply)

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    serve(args.port, args.log)
