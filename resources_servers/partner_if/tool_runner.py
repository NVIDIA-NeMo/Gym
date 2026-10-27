# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve one task's `ToolBackend` over HTTP inside its sandbox, for a sandboxed mock-tool server.

tool_sandbox.py uploads this file, tool_backend.py, the task's bundle and runner.json into a fresh sandbox and
runs `python3 -I tool_runner.py runner.json`. The runner uses only the standard library, so any Python 3 image
can run it, and it holds a single episode. A request without runner.json's token is refused.

The routes mirror the protocol. `GET /requires_final_answer` is also the readiness probe: it answers
`{"status": "loading"}` while the backend loads and `{"status": "failed", "error": ...}` if it could not.
`POST /tools` and `POST /new_state` take `{"row": ...}`. `POST /call` takes `{"name": ..., "arguments": ...}`
and answers `{"output": ...}`, or `{"tool_call_error": ...}` for a ToolCallError. `GET /artifacts` and
`GET /receipts` answer `{"artifacts": ...}` and `{"receipts": ...}`. Any other exception from the backend is a
500 whose `error` is the traceback, for the server's log; the server never shows it to the model.
"""

import hmac
import json
import re
import sys
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module, metadata
from pathlib import Path


TOKEN_HEADER = "X-Tool-Runner-Token"
# Directories beside runner.json: the task's files, its installed requirements, and the protocol module.
BUNDLE_DIR, SITE_DIR, LIB_DIR = "bundle", "site", "lib"


class Conflict(Exception):
    """A request out of order, such as a call before the episode's state exists."""


class Runner:
    """One episode of a task's backend: loaded once, then called one request at a time."""

    def __init__(self, config: dict, root: Path):
        self.config = config
        self.root = root
        self.lock = threading.Lock()
        self.loaded = threading.Event()
        self.load_error: str | None = None
        self.backend = None
        self.requires_final_answer = False
        self.tool_call_error: type[Exception] = Exception
        self.state = None
        self.has_state = False

    def load(self) -> None:
        try:
            for requirement in self.config["requirements"]:
                name, _, pinned = re.sub(r"\[.*\]", "", requirement).partition("==")
                try:
                    found = metadata.version(name)
                except metadata.PackageNotFoundError:
                    found = None
                if found != pinned:
                    raise RuntimeError(f"The backend needs {requirement}, but {name} is {found or 'not installed'}")
            self.tool_call_error = import_module("resources_servers.partner_if.tool_backend").ToolCallError
            module, _, name = self.config["entrypoint"].partition(":")
            self.backend = getattr(import_module(module), name)(self.root / BUNDLE_DIR / self.config["fixture"])
            self.requires_final_answer = bool(self.backend.requires_final_answer)
        except Exception:
            self.load_error = traceback.format_exc()
        finally:
            self.loaded.set()

    def handle(self, method: str, route: str, body: dict) -> tuple[HTTPStatus, dict]:
        if not self.loaded.is_set():
            return HTTPStatus.SERVICE_UNAVAILABLE, {"status": "loading"}
        if self.load_error is not None:
            return HTTPStatus.INTERNAL_SERVER_ERROR, {"status": "failed", "error": self.load_error}
        handler = ROUTES.get((method, route))
        if handler is None:
            return HTTPStatus.NOT_FOUND, {"error": f"No route {method} /{route}"}
        with self.lock:
            try:
                return HTTPStatus.OK, handler(self, body)
            except Conflict as error:
                return HTTPStatus.CONFLICT, {"error": str(error)}
            except self.tool_call_error as error:
                return HTTPStatus.OK, {"tool_call_error": str(error)}
            except Exception:
                return HTTPStatus.INTERNAL_SERVER_ERROR, {"error": traceback.format_exc()}

    def episode_state(self):
        if not self.has_state:
            raise Conflict("The episode has no state yet")
        return self.state

    def get_requires_final_answer(self, body: dict) -> dict:
        return {"status": "ready", "requires_final_answer": self.requires_final_answer}

    def tools(self, body: dict) -> dict:
        return {"tools": self.backend.tools(body["row"])}

    def new_state(self, body: dict) -> dict:
        if self.has_state:
            raise Conflict("The episode's state already exists")
        self.state = self.backend.new_state(body["row"])
        self.has_state = True
        return {}

    def call(self, body: dict) -> dict:
        return {"output": self.backend.call(self.episode_state(), body["name"], body["arguments"])}

    def artifacts(self, body: dict) -> dict:
        return {"artifacts": self.backend.artifacts(self.episode_state())}

    def receipts(self, body: dict) -> dict:
        return {"receipts": self.backend.receipts(self.episode_state())}


ROUTES = {
    ("GET", "requires_final_answer"): Runner.get_requires_final_answer,
    ("POST", "tools"): Runner.tools,
    ("POST", "new_state"): Runner.new_state,
    ("POST", "call"): Runner.call,
    ("GET", "artifacts"): Runner.artifacts,
    ("GET", "receipts"): Runner.receipts,
}


def handler_for(runner: Runner) -> type[BaseHTTPRequestHandler]:
    token = runner.config["token"].encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.respond("GET")

        def do_POST(self) -> None:
            self.respond("POST")

        def respond(self, method: str) -> None:
            if not hmac.compare_digest(self.headers.get(TOKEN_HEADER, "").encode(), token):
                status, payload = HTTPStatus.FORBIDDEN, {"error": "Missing or wrong runner token"}
            else:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(length) or b"{}") if method == "POST" else {}
                except ValueError as error:
                    status, payload = HTTPStatus.BAD_REQUEST, {"error": f"Invalid request body: {error}"}
                else:
                    status, payload = runner.handle(method, self.path.strip("/"), body)
            try:
                content = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            except (TypeError, ValueError):  # A backend result that is not a JSON value.
                status = HTTPStatus.INTERNAL_SERVER_ERROR
                content = json.dumps({"error": traceback.format_exc()}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    return Handler


def main(config_path: str) -> None:
    root = Path(config_path).resolve().parent
    config = json.loads(Path(config_path).read_text())
    # The task's files first, so its entrypoint resolves there; then its pinned requirements over the image's.
    sys.path[:0] = [str(root / BUNDLE_DIR), str(root / SITE_DIR), str(root / LIB_DIR)]
    runner = Runner(config, root)
    # Listen before loading, so the server can tell a backend that is still loading from one that failed.
    server = ThreadingHTTPServer((config["host"], config["port"]), handler_for(runner))
    threading.Thread(target=runner.load, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1])
