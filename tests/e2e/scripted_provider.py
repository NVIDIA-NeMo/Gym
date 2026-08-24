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

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class ProviderState:
    def __init__(self, events_path: Path) -> None:
        self.events_path = events_path
        self.lock = threading.Lock()

    def record(self, body: dict[str, Any]) -> None:
        with self.lock:
            with self.events_path.open("a", encoding="utf-8") as events_file:
                events_file.write(json.dumps(body) + "\n")


class ProviderServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], state: ProviderState) -> None:
        super().__init__(address, Handler)
        self.state = state


class Handler(BaseHTTPRequestHandler):
    server: ProviderServer

    def _write_json(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._write_json(200, {"status": "ok"})
            return
        if self.path == "/v1/models":
            self._write_json(
                200,
                {
                    "object": "list",
                    "data": [{"id": "scripted-model", "object": "model"}],
                },
            )
            return
        self._write_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._write_json(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(content_length))
        self.server.state.record(body)

        messages = body.get("messages", [])
        if any(message.get("role") == "tool" for message in messages):
            message = {
                "role": "assistant",
                "content": "The weather in San Francisco is cold.",
            }
            finish_reason = "stop"
            completion_tokens = 9
        else:
            advertised_tools = {tool.get("function", {}).get("name") for tool in body.get("tools", [])}
            if "get_weather" not in advertised_tools:
                self._write_json(400, {"error": "Gym did not forward the get_weather tool"})
                return
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "San Francisco"}),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
            completion_tokens = 7

        self._write_json(
            200,
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "scripted-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": completion_tokens,
                    "total_tokens": 10 + completion_tokens,
                },
            },
        )

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--events", type=Path, required=True)
    args = parser.parse_args()

    args.events.parent.mkdir(parents=True, exist_ok=True)
    args.events.unlink(missing_ok=True)
    ProviderServer((args.host, args.port), ProviderState(args.events)).serve_forever()


if __name__ == "__main__":
    main()
