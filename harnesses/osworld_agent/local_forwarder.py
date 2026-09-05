# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local reverse proxy for path-based OpenSandbox guest endpoints.

OSWorld's controllers accept only ``http://host:port/...``. OpenSandbox's
server-proxy endpoints instead look like
``http://gateway/v1/sandboxes/<id>/proxy/<port>`` and may require route
headers. This forwarder exposes an ephemeral loopback port, prepends the
upstream path, injects route headers, and carries Chrome CDP WebSocket
upgrades.

The implementation is adapted from Cell-1's validated
``resources_servers/osworld/local_forwarder.py``. It deliberately disables
ambient HTTP proxy variables: this bridge must reach the OpenSandbox gateway,
not a workstation proxy.
"""

from __future__ import annotations

import re
import socket
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import requests


_HOP_BY_HOP_REQUEST_HEADERS = (
    "host",
    "content-length",
    "connection",
    "accept-encoding",
)
_HOP_BY_HOP_RESPONSE_HEADERS = (
    "transfer-encoding",
    "content-length",
    "connection",
    "content-encoding",
)

# Chrome's /json/version and /json/list responses embed an absolute CDP
# WebSocket URL. Rewrite it to the local bridge so the subsequent Upgrade
# request follows the same OpenSandbox proxy path.
_WS_URL_RE = re.compile(rb"wss?://[^/\"]+/")


def _open_upstream_socket(
    *,
    scheme: str,
    host: str,
    port: int,
) -> socket.socket:
    upstream = socket.create_connection((host, port), timeout=30)
    if scheme == "https":
        context = ssl.create_default_context()
        upstream = context.wrap_socket(upstream, server_hostname=host)
    upstream.settimeout(None)
    return upstream


def start_forwarder(
    base_url: str,
    extra_headers: dict[str, str] | None = None,
    timeout_s: float = 300.0,
) -> tuple[ThreadingHTTPServer, int]:
    """Start ``127.0.0.1:<ephemeral>/<path> -> <base_url>/<path>``.

    The caller owns the returned server and must call both ``shutdown()`` and
    ``server_close()`` during sandbox cleanup.
    """

    base = base_url.rstrip("/")
    headers_to_add = dict(extra_headers or {})
    split = urlsplit(base)
    if split.scheme not in {"http", "https"} or split.hostname is None:
        raise ValueError(f"forwarder requires an absolute HTTP(S) URL, got {base_url!r}")

    proxy_host = split.hostname
    proxy_port = split.port or (443 if split.scheme == "https" else 80)
    proxy_authority = split.netloc
    proxy_path_prefix = split.path.rstrip("/")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:
            del args

        def _tunnel_websocket(self) -> None:
            upstream = _open_upstream_socket(
                scheme=split.scheme,
                host=proxy_host,
                port=proxy_port,
            )
            lines = [
                f"GET {proxy_path_prefix}{self.path} HTTP/1.1",
                f"Host: {proxy_authority}",
            ]
            client_keys = {key.lower() for key in self.headers}
            for key, value in self.headers.items():
                if key.lower() != "host":
                    lines.append(f"{key}: {value}")
            for key, value in headers_to_add.items():
                if key.lower() not in client_keys:
                    lines.append(f"{key}: {value}")
            upstream.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
            client = self.connection

            def pump(source: socket.socket, destination: socket.socket) -> None:
                try:
                    while chunk := source.recv(65536):
                        destination.sendall(chunk)
                except OSError:
                    pass
                finally:
                    for stream in (source, destination):
                        try:
                            stream.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass

            upstream_to_client = threading.Thread(
                target=pump,
                args=(upstream, client),
                daemon=True,
            )
            upstream_to_client.start()
            pump(client, upstream)
            upstream_to_client.join(timeout=5)
            self.close_connection = True

        def _forward(self) -> None:
            if (self.headers.get("Upgrade") or "").lower() == "websocket":
                self._tunnel_websocket()
                return

            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length else None
            headers = {
                key: value for key, value in self.headers.items() if key.lower() not in _HOP_BY_HOP_REQUEST_HEADERS
            }
            # Guest authentication headers (for example VLC Basic auth) win
            # over route headers with the same name.
            lower_client_headers = {key.lower() for key in headers}
            for key, value in headers_to_add.items():
                if key.lower() not in lower_client_headers:
                    headers[key] = value

            try:
                with requests.Session() as session:
                    session.trust_env = False
                    upstream = session.request(
                        self.command,
                        base + self.path,
                        data=body,
                        headers=headers,
                        timeout=timeout_s,
                        allow_redirects=False,
                    )
            except Exception as error:  # noqa: BLE001
                message = str(error).encode()
                self.send_response(502)
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)
                return

            content = b"" if self.command == "HEAD" else upstream.content
            if b"webSocketDebuggerUrl" in content or b"webSocketUrl" in content:
                local = f"ws://127.0.0.1:{self.server.server_address[1]}/".encode()
                content = _WS_URL_RE.sub(local, content)

            self.send_response(upstream.status_code)
            for key, value in upstream.headers.items():
                if key.lower() not in _HOP_BY_HOP_RESPONSE_HEADERS:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            if content:
                self.wfile.write(content)

        do_DELETE = _forward
        do_GET = _forward
        do_HEAD = _forward
        do_OPTIONS = _forward
        do_PATCH = _forward
        do_POST = _forward
        do_PUT = _forward

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = int(server.server_address[1])
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, port
