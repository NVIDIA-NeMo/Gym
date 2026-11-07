# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stub Gym Model Server for crews that call the model over HTTP, from the sandbox runner, and a lendable sandbox."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

from nemo_gym.sandbox import get_provider_class


class LendableLocalProvider(get_provider_class("local")):
    """The local provider, made connectable by sandbox id so a test can lend its sandbox to the agent, as a
    Resources Server lends a Docker or OpenSandbox sandbox. Records each provider close of a sandbox."""

    name = "lendable_local"
    handles: ClassVar[dict] = {}
    closed: ClassVar[list] = []

    async def serialize_handle(self, handle, *, scope=None) -> dict:
        self.handles[handle.sandbox_id] = handle
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor):
        return self.handles[descriptor["sandbox_id"]]

    async def close(self, handle) -> None:
        self.closed.append(handle.sandbox_id)
        await super().close(handle)


USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
# Not part of the response body: the stub sends these as Set-Cookie headers.
_SET_COOKIES = "_set_cookies"


def completion(
    content: str | None = None,
    *calls: tuple[str, dict],
    id: str | None,
    usage: bool = True,
    finish_reason: str = "stop",
    cookies: dict[str, str] | None = None,
) -> dict:
    """A Chat Completions response with the assistant's text and tool calls, and the cookies the server sets."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {"id": f"{id}-call-{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(calls)
        ]
    body: dict[str, Any] = {"choices": [{"index": 0, "message": message, "finish_reason": finish_reason}]}
    if id is not None:
        body["id"] = id
    if usage:
        body["usage"] = USAGE
    if cookies:
        body[_SET_COOKIES] = cookies
    return body


class ModelServer:
    """Answer Chat Completions requests in order and record each request's path, headers and body.

    An answer is a response body, an ``int`` HTTP error status, or ``None``, which hangs until the server closes.
    Once the answers run out, the last one answers every later request, as a server that keeps rejecting does.
    """

    def __init__(self, answers: list[dict | int | tuple[int, dict] | tuple[int, dict, dict[str, str]] | None]) -> None:
        self.answers = list(answers)
        self.requests: list[dict[str, Any]] = []
        self.blocked = threading.Event()
        self.release = threading.Event()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append(
                    {"path": self.path, "headers": dict(self.headers), "json": body, "received_at": time.monotonic()}
                )
                answer = server.answers[min(len(server.requests), len(server.answers)) - 1]
                if answer is None:
                    server.blocked.set()
                    server.release.wait(60)
                    return
                headers = {}
                if isinstance(answer, tuple):
                    status, answer, *extra = answer
                    if extra:
                        [headers] = extra
                elif isinstance(answer, int):
                    status, answer = answer, {"error": {"message": "rejected"}}
                else:
                    status, answer = 200, dict(answer)
                answer = dict(answer)
                cookies = answer.pop(_SET_COOKIES, {})
                payload = json.dumps(answer).encode()
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                for name, value in cookies.items():
                    self.send_header("Set-Cookie", f"{name}={value}; Path=/")
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
        """The server's URL as ``resolve_model_base_url()`` gives it, with the rollout prefix and ending in ``/v1``."""
        return f"http://127.0.0.1:{self._server.server_address[1]}/ng-rollout/rollout/v1"

    def close(self) -> None:
        self.release.set()
        self._server.shutdown()
        self._server.server_close()
