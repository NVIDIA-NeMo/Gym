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
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from http.client import HTTPConnection, HTTPException, HTTPSConnection, IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit
from uuid import uuid4


MAX_HISTORY_BODY_BYTES = 32 * 1024 * 1024
_MAX_HISTORY_STATES = 64
_CONVERSATION_HEADERS = ("thread-id", "x-session-affinity", "x-opencode-session", "x-session-id")
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


def _dump_opencode_serialized_member(value: dict[str, Any], member: str, serialized: str) -> str:
    return (
        "{"
        + ",".join(
            json.dumps(key) + ":" + (serialized if key == member else json.dumps(item)) for key, item in value.items()
        )
        + "}"
    )


def _serialize_opencode_replacement_config(config: dict[str, Any]) -> str:
    """Keep only the replacement prompt literal through OpenCode's pre-JSON substitutions."""
    # Keep in sync with opencode_agent.app._serialize_replacement_config.
    # This standalone sandbox helper cannot import the Gym agent application.
    agent = config["agent"]
    build = agent["build"]
    prompt = json.dumps(build["prompt"]).replace("{", "\\u007b")
    build_json = _dump_opencode_serialized_member(build, "prompt", prompt)
    agent_json = _dump_opencode_serialized_member(agent, "build", build_json)
    return _dump_opencode_serialized_member(config, "agent", agent_json)


class DeltaState:
    """Track an optimistic history base shared by concurrent model requests."""

    def __init__(self) -> None:
        self.conversation_id = uuid4().hex
        self.version: str | None = None
        self.items: list[bytes] = []
        self._lock = threading.Lock()

    def prepare(self, body: dict, field: str, *, full: bool = False) -> tuple[bytes, str, list[bytes]]:
        items = [_encoded(item) for item in body[field]]
        history_bytes = b"[" + b",".join(items) + b"]"
        with self._lock:
            version, previous_items = self.version, self.items
        # submitted() replaces the list without mutating published lists, so
        # this snapshot remains consistent while another request is submitted.
        append = not full and version is not None and items[: len(previous_items)] == previous_items
        identity = uuid4().hex
        envelope = {
            "conversation_id": self.conversation_id,
            "request_id": identity,
            "base_version": version if append else None,
            "mode": "append" if append else "full",
            "messages_sha256": hashlib.sha256(history_bytes).hexdigest(),
        }
        payload = {
            **body,
            field: body[field][len(previous_items) :] if append else body[field],
            "_gym_history": envelope,
        }
        return _encoded(payload), identity, items

    def submitted(self, identity: str, items: list[bytes]) -> None:
        """Keep a candidate base after sending; Gym rejects it if it was not stored."""
        with self._lock:
            self.version, self.items = identity, items


class HistoryTransport:
    """Forward parallel exchanges through a bounded, reusable connection pool.

    A CLI may issue concurrent subagent calls. Each exchange owns its upstream
    connection until its response body has been consumed and forwarded, or fails.
    At most ``max_connections`` upstream connections are active or cached. Extra
    requests wait for a slot, using ``timeout`` as the pool-wait deadline.
    History versions remain optimistic: a stale base triggers Gym's explicit resync.
    """

    def __init__(self, base_url: str, *, timeout: float, max_connections: int = 8) -> None:
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
        if isinstance(max_connections, bool) or not isinstance(max_connections, int) or max_connections < 1:
            raise ValueError("max_connections must be a positive integer")
        self.timeout = timeout
        self._max_connections = max_connections
        self._states: OrderedDict[tuple[str, str, str], DeltaState] = OrderedDict()
        self._lock = threading.Lock()
        self._connection_available = threading.Condition(self._lock)
        self._connections: set[HTTPConnection] = set()
        self._idle_connections: list[HTTPConnection] = []
        self._client_sockets: set[socket.socket] = set()
        self._closed = False
        cls = HTTPSConnection if self.target.scheme == "https" else HTTPConnection
        owner = self

        class Connection(cls):
            def connect(self) -> None:
                with owner._lock:
                    if owner._closed:
                        raise OSError("History transport is closing")
                super().connect()
                # Shutdown may start while TCP/TLS connection setup is blocked.
                # Never send a request on a socket that completed setup afterward.
                with owner._lock:
                    if owner._closed:
                        self.close()
                        raise OSError("History transport is closing")

        self._connection_class = Connection

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(owner.timeout)
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with owner._lock:
                    closed = owner._closed
                    if not closed:
                        owner._client_sockets.add(self.connection)
                if closed:
                    owner._abort_socket(self.connection)

            def finish(self) -> None:
                try:
                    super().finish()
                finally:
                    with owner._lock:
                        owner._client_sockets.discard(self.connection)

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
        # Under the checkout lock, block new leases and wake existing waiters.
        # Both reject checkout after observing _closed; only prior leases remain.
        with self._connection_available:
            self._closed = True
            self._connection_available.notify_all()
            connections = list(self._connections)
            idle_connections = list(self._idle_connections)
            self._idle_connections.clear()
            self._connections.difference_update(idle_connections)
            client_sockets = list(self._client_sockets)
            self._states.clear()
        # Abort socket I/O, rather than concurrently mutating an active
        # HTTPConnection's response state with close(). Its lease closes it.
        for connection in connections:
            upstream_socket = connection.sock
            if upstream_socket is not None:
                self._abort_socket(upstream_socket)
        for client_socket in client_sockets:
            self._abort_socket(client_socket)
        for connection in idle_connections:
            connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)

    @staticmethod
    def _abort_socket(connection: socket.socket) -> None:
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # Already disconnected, or never connected.

    @staticmethod
    def _history_identity(headers: Any) -> tuple[str, str]:
        # Prefer the harness's conversation identity over Gym's invocation-level
        # x-session-id. Unknown clients retain the endpoint-level fallback.
        return next(
            ((name, headers[name]) for name in _CONVERSATION_HEADERS if headers.get(name)),
            ("", ""),
        )

    def _history_state(self, path: str, headers: Any) -> DeltaState:
        key = (path, *self._history_identity(headers))
        with self._lock:
            if self._closed:
                raise OSError("History transport is closing")
            state = self._states.get(key)
            if state is None:
                state = self._states[key] = DeltaState()
            self._states.move_to_end(key)
            while len(self._states) > _MAX_HISTORY_STATES:
                # No reliable subagent-exit signal is available. Evict the least
                # recently used cache entry; its next call starts a full history.
                # In-flight calls retain their own reference and remain valid.
                self._states.popitem(last=False)
            return state

    @contextmanager
    def _upstream_connection(self) -> Iterator[HTTPConnection]:
        """Lease one connection for a complete exchange; discard it on any failure."""
        with self._connection_available:
            available = self._connection_available.wait_for(
                lambda: self._closed or self._idle_connections or len(self._connections) < self._max_connections,
                timeout=self.timeout,
            )
            if not available:
                raise TimeoutError("Timed out waiting for a history transport connection")
            if self._closed:
                raise OSError("History transport is closing")
            if self._idle_connections:
                connection = self._idle_connections.pop()
            else:
                connection = self._connection_class(self.target.hostname, self.target.port, timeout=self.timeout)
                self._connections.add(connection)
        reusable = False
        try:
            yield connection
            reusable = True
        finally:
            with self._connection_available:
                keep_idle = reusable and not self._closed
                if keep_idle:
                    self._idle_connections.append(connection)
                    self._connection_available.notify()
            if not keep_idle:
                try:
                    connection.close()
                finally:
                    with self._connection_available:
                        self._connections.discard(connection)
                        self._connection_available.notify()

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
        with self._lock:
            if self._closed:
                handler.close_connection = True
                return
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
                    state = self._history_state(path.path, handler.headers)
                    prepared = state.prepare(body, field)
                    wire = prepared[0]
            connection_headers = {value.strip().lower() for value in handler.headers.get("Connection", "").split(",")}
            headers = {
                key: value for key, value in handler.headers.items() if key.lower() not in _HOP | connection_headers
            }
            headers["x-gym-client-request-id"] = identity
            if prepared:
                headers["x-gym-history-transport"] = "v1"
            with self._upstream_connection() as connection:
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
            if has_body:
                # Return the consumed upstream connection before publishing completion
                # so the next sequential request can reuse it from any CLI connection.
                handler.wfile.write(b"0\r\n\r\n")
                handler.wfile.flush()
        except (OSError, HTTPException, ValueError, TypeError) as error:
            # Do not retry disconnects, timeouts, model failures, or stream errors.
            # The native client retains its own error/retry policy.
            handler.close_connection = True
            with self._lock:
                if self._closed:
                    return  # Shutdown intentionally aborted both sides of this exchange.
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
        params["env"]["OPENCODE_CONFIG_CONTENT"] = (
            _serialize_opencode_replacement_config(config)
            if params.get("literal_replacement_prompt", False)
            else json.dumps(config)
        )
    elif harness == "pi":
        path = directory / "home/.pi/agent/models.json"
        config = json.loads(path.read_text())
        config["providers"]["nemo"]["baseUrl"] = local_url
        path.write_text(json.dumps(config))
    elif harness == "codex":
        # Insert before the prompt separator; only this invocation's provider changes.
        index = params["command"].index("--")
        params["command"][index:index] = ["-c", "model_providers.gym.base_url=" + json.dumps(local_url)]
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
