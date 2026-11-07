# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stub Gym Model Server for runs whose graph calls the model over HTTP, from the sandbox runner."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


USAGE = {
    "input_tokens": 10,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 5,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 15,
}


def call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    """A function_call output item."""
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(args), "id": call_id}


def text(content: str) -> dict[str, Any]:
    """An assistant message output item."""
    return {
        "type": "message",
        "id": "msg",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": content, "annotations": []}],
    }


class ModelServer:
    """Answer Responses API requests in order and record each request body and path.

    An answer is a list of output items, an ``int`` HTTP error status, or ``None``, which hangs until closed.
    """

    def __init__(
        self, answers: list[list[dict[str, Any]] | int | tuple[int, dict] | None], *, usage: dict | None = USAGE
    ) -> None:
        self.answers = answers
        self.requests: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self.blocked = threading.Event()
        self.release = threading.Event()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                server.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                server.paths.append(self.path)
                answer = server.answers[len(server.requests) - 1]
                if answer is None:
                    server.blocked.set()
                    server.release.wait(60)
                    answer = [text("late")]
                if isinstance(answer, tuple):
                    status, body = answer
                elif isinstance(answer, int):
                    status, body = answer, {"error": {"message": "rejected", "type": "stub_error"}}
                else:
                    status = 200
                    body = {
                        "id": f"resp_{len(server.requests)}",
                        "created_at": 0,
                        "model": "policy",
                        "object": "response",
                        "output": answer,
                        "parallel_tool_calls": False,
                        "tool_choice": "auto",
                        "tools": [],
                        "usage": usage,
                    }
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        """The server's URL as `resolve_model_base_url()` gives it, ending in `/v1`."""
        return f"http://127.0.0.1:{self._server.server_address[1]}/ng-rollout/rollout/v1"

    def close(self) -> None:
        self.release.set()
        self._server.shutdown()
        self._server.server_close()
