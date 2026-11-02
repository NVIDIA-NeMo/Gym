# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox-local, stdlib-only history transport for unmodified native CLI clients.

This loopback helper runs only when history deltas are enabled.
SSE bytes are relayed immediately, without interpreting model output.
This file is uploaded beside sandbox_runner.py; it cannot import Gym dependencies.
"""

from __future__ import annotations

import hashlib
import json
import socket
import threading
from contextlib import ExitStack
from http.client import HTTPConnection, HTTPException, HTTPSConnection, IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4


MAX_HISTORY_BODY_BYTES = 32 * 1024 * 1024
_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "expect",
}


def _encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class DeltaState:
    """Track submitted history for one sequential conversation."""

    def __init__(self) -> None:
        self.conversation_id = uuid4().hex
        self.version: str | None = None
        self.items: list[bytes] = []

    def prepare(self, body: dict, field: str, *, full: bool = False) -> tuple[bytes, str, list[bytes]]:
        items = [_encoded(item) for item in body[field]]
        history_bytes = b"[" + b",".join(items) + b"]"
        append = not full and self.version is not None and items[: len(self.items)] == self.items
        identity = uuid4().hex
        envelope = {
            "conversation_id": self.conversation_id,
            "request_id": identity,
            "base_version": self.version if append else None,
            "mode": "append" if append else "full",
            "messages_sha256": hashlib.sha256(history_bytes).hexdigest(),
        }
        payload = {
            **body,
            field: body[field][len(self.items) :] if append else body[field],
            "_gym_history": envelope,
        }
        return _encoded(payload), identity, items

    def submitted(self, identity: str, items: list[bytes]) -> None:
        """Keep a candidate base after sending; Gym rejects it if it was not stored."""
        self.version, self.items = identity, items


class HistoryTransport:
    """One loopback listener and reusable Gym connection per sequential invocation."""

    def __init__(self, base_url: str, *, timeout: float) -> None:
        self.target = urlsplit(base_url.rstrip("/"))
        if (
            self.target.scheme not in {"http", "https"}
            or not self.target.hostname
            or self.target.username
            or self.target.password
            or self.target.query
            or self.target.fragment
        ):
            raise ValueError("Invalid history transport endpoint")
        self.timeout = timeout
        self._states: dict[str, DeltaState] = {}
        cls = HTTPSConnection if self.target.scheme == "https" else HTTPConnection
        self._connection = cls(self.target.hostname, self.target.port, timeout=self.timeout)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(owner.timeout)
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            def log_message(self, *_: object) -> None:
                pass  # Never log request paths, credentials, or model content.

            def do_POST(self) -> None:
                owner.handle(self)

            def do_GET(self) -> None:
                owner.handle(self)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}{self.target.path}"

    def __exit__(self, *_: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self._connection.close()
        self.thread.join(timeout=1)

    @staticmethod
    def read_request(handler: BaseHTTPRequestHandler) -> bytes:
        if handler.headers.get("Transfer-Encoding", "").lower() == "chunked":
            chunks = bytearray()
            while True:
                line = handler.rfile.readline(128)
                size = int(line.split(b";", 1)[0].strip(), 16)
                if size < 0 or len(chunks) + size > MAX_HISTORY_BODY_BYTES:
                    raise ValueError("Request too large")
                if size == 0:
                    if handler.rfile.readline(4) != b"\r\n":
                        raise ValueError("Request trailers are unsupported")
                    return bytes(chunks)
                chunk = handler.rfile.read(size)
                if len(chunk) != size or handler.rfile.read(2) != b"\r\n":
                    raise ValueError("Incomplete chunk")
                chunks.extend(chunk)
        if handler.headers.get("Transfer-Encoding"):
            raise ValueError("Unsupported transfer encoding")
        size = int(handler.headers.get("Content-Length", "0"))
        if not 0 <= size <= MAX_HISTORY_BODY_BYTES:
            raise ValueError("Request too large")
        body = handler.rfile.read(size)
        if len(body) != size:
            raise ValueError("Incomplete request")
        return body

    def handle(self, handler: BaseHTTPRequestHandler) -> None:
        connection = None
        response_started = False
        identity = handler.headers.get("x-gym-client-request-id", "")
        if len(identity) != 32 or any(c not in "0123456789abcdef" for c in identity):
            identity = uuid4().hex
        try:
            # Retain the rollout/capture prefix. Never accept an absolute proxy URL
            # or let a CLI change the upstream host through a request header.
            path = urlsplit(handler.path)
            if path.scheme or path.netloc or not path.path.startswith(self.target.path + "/"):
                raise ValueError("Unexpected model path")
            if handler.headers.get("Upgrade") or handler.headers.get("x-gym-history-transport"):
                raise ValueError("Unsupported transport")
            original = self.read_request(handler)
            field = (
                "messages"
                if path.path == self.target.path + "/chat/completions"
                else "input"
                if path.path == self.target.path + "/responses"
                else None
            )
            body, state, prepared = None, None, None
            wire = original
            if (
                handler.command == "POST"
                and field
                and handler.headers.get("Content-Encoding", "identity") == "identity"
            ):
                body = json.loads(original)
                if isinstance(body, dict) and "_gym_history" in body:
                    raise ValueError("Reserved history field")
                if (
                    isinstance(body, dict)
                    and isinstance(body.get(field), list)
                    and all(isinstance(item, dict) for item in body[field])
                ):
                    state = self._states.setdefault(path.path, DeltaState())
                    prepared = state.prepare(body, field)
                    wire = prepared[0]
            # Reuse the same Gym connection even when the CLI opens a new local
            # connection. Requests within this rollout must remain sequential.
            # http.client preserves compressed/SSE data while handling framing.
            connection = self._connection
            connection_headers = {value.strip().lower() for value in handler.headers.get("Connection", "").split(",")}
            headers = {
                key: value for key, value in handler.headers.items() if key.lower() not in _HOP | connection_headers
            }
            headers["x-gym-client-request-id"] = identity
            if prepared:
                headers["x-gym-history-transport"] = "v1"
            for attempt in range(2):
                headers["Content-Length"] = str(len(wire))
                connection.request(handler.command, handler.path, body=wire, headers=headers)
                if prepared:
                    # Gym stores history before inference. A lost response does not
                    # imply a lost base; let the next request try that version.
                    # A missing base still triggers the explicit 409/full fallback.
                    state.submitted(prepared[1], prepared[2])
                response = connection.getresponse()
                if response.status == 409 and prepared and attempt == 0:
                    rejection = response.read(MAX_HISTORY_BODY_BYTES + 1)
                    try:
                        detail = json.loads(rejection).get("error", {})
                    except (ValueError, AttributeError):
                        detail = {}
                    if isinstance(detail, dict) and detail.get("code") == "gym_history_resync":
                        prepared = state.prepare(body, field, full=True)
                        wire = prepared[0]
                        continue  # Only Gym's explicit pre-inference rejection is retried.
                    buffered = rejection
                else:
                    buffered = None
                break
            with response:
                handler.send_response(response.status)
                response_hop = {value.strip().lower() for value in response.getheader("Connection", "").split(",")}
                for key, value in response.getheaders():
                    if key.lower() not in _HOP | response_hop:
                        handler.send_header(key, value)
                has_body = response.status not in {204, 304} and response.status >= 200
                if has_body:
                    handler.send_header("Transfer-Encoding", "chunked")
                handler.end_headers()
                response_started = True
                if has_body:
                    while True:
                        chunk = buffered if buffered is not None else response.read1(64 * 1024)
                        buffered = None
                        if not chunk:
                            if response.length:
                                raise IncompleteRead(b"", response.length)
                            break
                        handler.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                        handler.wfile.flush()
                    handler.wfile.write(b"0\r\n\r\n")
                    handler.wfile.flush()
        except (OSError, HTTPException, ValueError, TypeError) as error:
            # Do not retry disconnects, timeouts, model failures, or stream errors.
            # The native client retains its own error/retry policy.
            handler.close_connection = True
            if connection is not None:
                connection.close()
            if not response_started:
                code = 400 if isinstance(error, (ValueError, TypeError)) else 502
                handler.send_error(code, "History transport request failed")


def configure_transport(stack: ExitStack, params: dict, *, harness: str) -> None:
    """Rewrite only a native CLI's Gym provider URL, leaving model settings intact."""
    transport = params.get("history_transport")
    if transport is None:
        return
    directory = Path(params["directory"])
    local_url = stack.enter_context(
        HistoryTransport(
            transport["base_url"],
            timeout=transport["timeout"],
        )
    )
    if harness == "opencode":
        config = json.loads(params["env"]["OPENCODE_CONFIG_CONTENT"])
        config["provider"]["nemo_gym"]["options"]["baseURL"] = local_url
        params["env"]["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
    elif harness == "pi":
        path = directory / "home/.pi/agent/models.json"
        config = json.loads(path.read_text())
        config["providers"]["nemo"]["baseUrl"] = local_url
        path.write_text(json.dumps(config))
    elif harness == "codex":
        # Keep the upstream Node launcher's 429 proxy and environment handling.
        # Its proxy targets this helper, which alone changes the history wire format.
        path = directory / "input.json"
        config = json.loads(path.read_text())
        config["model_base_url"] = local_url
        path.write_text(json.dumps(config))
    else:
        raise ValueError("Unknown native history transport harness")


async def stage_cli_transport(sandbox: Any, params: dict, *, harness: str) -> list[str]:
    """Wrap an upstream Bash runner's CLI argv only when delta transport is selected."""
    if params.get("history_transport") is None:
        return params["command"]
    from nemo_gym.sandbox.utils import upload_text

    directory = params["directory"]
    await sandbox.upload(Path(__file__), f"{directory}/history_transport")
    path = f"{directory}/history-transport-input.json"
    await upload_text(sandbox, path=path, text=json.dumps(params))
    # Match upstream's neutral runner naming: task cleanup such as pkill -f
    # python3 or .py must not terminate the transport while its CLI still runs.
    from shlex import quote

    resolve = quote("import sys; print(getattr(sys, '_base_executable', sys.executable))")
    launcher = (
        f"real=$(python3 -I -c {resolve}) || exit 125; "
        'mkdir -p "$1/bin" && ln -sfn "$real" "$1/bin/gym-history-runner" || exit 125; '
        'exec "$1/bin/gym-history-runner" -I "$1/history_transport" "$2" "$3"'
    )
    return ["bash", "-c", launcher, "history-transport", directory, path, harness]


def main() -> None:
    """Keep the helper alive around one CLI; upstream runners retain artifact/supervisor ownership."""
    import os
    import signal
    import subprocess
    import sys

    params = json.loads(Path(sys.argv[1]).read_text())
    child = None
    stopping = False

    def interrupt(*_: object) -> None:
        nonlocal stopping
        stopping = True
        if child is not None:
            child.terminate()

    signal.signal(signal.SIGTERM, interrupt)
    with ExitStack() as stack:
        configure_transport(stack, params, harness=sys.argv[2])
        child = subprocess.Popen(params["command"], cwd=params["cwd"], env={**os.environ, **params["env"]})
        try:
            if stopping:
                child.terminate()
            code = child.wait()
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
