# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stub model server and a resources app served over HTTP, shared by the runner and session tests."""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel


_TOOLS = json.loads((Path(__file__).resolve().parents[3] / "benchmarks" / "gdpval" / "tools.json").read_text())


def _call(name: str, arguments: str) -> dict:
    tool_call = {"id": f"call-{name}", "type": "function", "function": {"name": name, "arguments": arguments}}
    return {"role": "assistant", "content": None, "tool_calls": [tool_call]}


class _ModelServer:
    """Answer chat completions in order and record each request body.

    A ``None`` answer hangs until closed; an ``int`` answer is an HTTP error status.
    """

    def __init__(self, answers: list[dict | int | None]) -> None:
        self.answers = answers
        self.blocked = threading.Event()
        self.release = threading.Event()
        self.requests: list[dict] = []
        self.session_ids: list[str | None] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                server.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                server.session_ids.append(self.headers.get("x-session-id"))
                message = server.answers[len(server.requests) - 1]
                if isinstance(message, int):
                    body = json.dumps({"error": {"message": "rejected", "type": "stub_error"}}).encode()
                    self.send_response(message)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if message is None:
                    server.blocked.set()
                    server.release.wait(60)
                    message = {"role": "assistant", "content": "late"}
                payload = json.dumps(
                    {
                        "id": f"chatcmpl-{len(server.requests)}",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "policy",
                        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.release.set()
        self._server.shutdown()
        self._server.server_close()


class _Finish(BaseModel):
    reason: str
    paths: list[str] = []


class _Search(BaseModel):
    query: str


def _resources_app(calls: list[tuple[str, dict, dict]]) -> FastAPI:
    """Routes that answer like the GDPVal resources server: a JSON string, a 400 detail, a 422 or a 500."""
    app = FastAPI()

    @app.post("/web_search")
    async def web_search(body: _Search, request: Request, response: Response) -> str:
        calls.append(("web_search", body.model_dump(), dict(request.cookies)))
        response.set_cookie("session", "after-search")
        return f"results for {body.query}"

    @app.post("/finish")
    async def finish(body: _Finish, request: Request) -> str:
        calls.append(("finish", body.model_dump(), dict(request.cookies)))
        if body.reason == "unprocessable":
            raise HTTPException(422, "reason is not accepted")
        if "missing.txt" in body.paths:
            raise HTTPException(400, "ERROR: Files do not exist: ['missing.txt']")
        return body.reason

    @app.post("/fetch_web_page")
    async def fetch_web_page() -> str:
        raise HTTPException(500, "boom")

    return app


class _Resources:
    def __init__(self, app: FastAPI) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self._server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            assert time.monotonic() < deadline, "resources app did not start"
            time.sleep(0.01)

    def close(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)
