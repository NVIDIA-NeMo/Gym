# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Local reverse-proxy forwarder for path-based sandbox proxies.

The upstream OSWorld controllers and evaluator getters build ``http://{ip}:{port}/...``
URLs, which can express neither a path-based proxy base
(``http://<domain>/sandboxes/<id>/proxy/<port>``) nor per-request route headers some
deployments require. A forwarder gives them a plain ``127.0.0.1:<port>`` that
transparently maps every request onto the proxied base and injects the headers.

Synchronous (``requests`` + ``ThreadingHTTPServer``) by design: it runs inside the
short-lived ``eval_task.py`` subprocess next to the synchronous upstream harness, never
inside the async server process.
"""

import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import requests


_HOP_BY_HOP_REQUEST_HEADERS = ("host", "content-length", "connection", "accept-encoding")
_HOP_BY_HOP_RESPONSE_HEADERS = ("transfer-encoding", "content-length", "connection", "content-encoding")

# Chrome CDP discovery responses (/json/version, /json/list) embed absolute
# ws://<guest-ip>:<port>/... URLs. Clients (playwright connect_over_cdp) dial that address
# verbatim, bypassing the forwarder — rewrite it to the forwarder's own listen address so the
# subsequent WebSocket CONNECT also flows through the path proxy (which carries upgrades).
_WS_URL_RE = re.compile(rb"ws://[^/\"]+/")


def start_forwarder(
    base_url: str,
    extra_headers: dict[str, str] | None = None,
    timeout_s: float = 300.0,
) -> tuple[ThreadingHTTPServer, int]:
    """Start a daemon-thread forwarder ``127.0.0.1:<ephemeral>/<path> -> <base_url>/<path>``.

    Returns the server (call ``shutdown()`` to stop) and the bound local port.
    """
    base = base_url.rstrip("/")
    headers_to_add = dict(extra_headers or {})

    split = urlsplit(base)
    proxy_host = split.hostname or "127.0.0.1"
    proxy_port = split.port or (443 if split.scheme == "https" else 80)
    proxy_path_prefix = split.path.rstrip("/")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence per-request logging
            del args

        def _tunnel_websocket(self):
            """Relay a WebSocket connection through the path proxy as raw TCP after replaying
            the client's Upgrade handshake against the proxied path. The OpenSandbox proxy
            carries Upgrade end-to-end (probe-verified), so a byte pump suffices."""
            upstream = socket.create_connection((proxy_host, proxy_port), timeout=30)
            lines = [f"GET {proxy_path_prefix}{self.path} HTTP/1.1", f"Host: {proxy_host}"]
            client_keys = {key.lower() for key in self.headers.keys()}
            for key, value in self.headers.items():
                if key.lower() == "host":
                    continue
                lines.append(f"{key}: {value}")
            for key, value in headers_to_add.items():
                if key.lower() not in client_keys:
                    lines.append(f"{key}: {value}")
            upstream.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
            client = self.connection

            def pump(src, dst):
                try:
                    while True:
                        chunk = src.recv(65536)
                        if not chunk:
                            break
                        dst.sendall(chunk)
                except OSError:
                    pass
                finally:
                    for sock in (src, dst):
                        try:
                            sock.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass

            t = threading.Thread(target=pump, args=(upstream, client), daemon=True)
            t.start()
            pump(client, upstream)
            t.join(timeout=5)
            self.close_connection = True

        def _forward(self):
            if (self.headers.get("Upgrade") or "").lower() == "websocket":
                self._tunnel_websocket()
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length else None
            headers = {
                key: value for key, value in self.headers.items() if key.lower() not in _HOP_BY_HOP_REQUEST_HEADERS
            }
            # Client-sent headers win: guest services authenticate too (e.g. VLC's http
            # interface Basic auth), and clobbering Authorization with the proxy's key
            # turns every such getter into a silent 401.
            lower_client = {key.lower() for key in headers}
            for key, value in headers_to_add.items():
                if key.lower() not in lower_client:
                    headers[key] = value
            try:
                upstream = requests.request(
                    self.command, base + self.path, data=body, headers=headers, timeout=timeout_s
                )
            except Exception as e:  # noqa: BLE001 - surface transport failures as 502 to the caller
                message = str(e).encode()
                self.send_response(502)
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)
                return
            content = upstream.content
            if b"webSocketDebuggerUrl" in content or b"webSocketUrl" in content:
                local = ("ws://127.0.0.1:%d/" % self.server.server_address[1]).encode()
                content = _WS_URL_RE.sub(local, content)
            self.send_response(upstream.status_code)
            for key, value in upstream.headers.items():
                if key.lower() in _HOP_BY_HOP_RESPONSE_HEADERS:
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        do_GET = _forward
        do_POST = _forward
        do_PUT = _forward
        do_DELETE = _forward

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, port
