# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-local Streamable HTTP transport for the existing Claw media tools.

No model endpoint is implemented here. Native N4 harnesses receive an ordinary
required MCP HTTP grant; media decoding and budgets remain MediaServer's.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


if __package__:
    from .mcp_server import MediaServer
else:
    from mcp_server import MediaServer

PROTOCOLS = {"2024-11-05", "2025-03-26", "2025-06-18"}
HOST = "127.0.0.1"
PORT = 17654
MAX_REQUEST_BYTES = 1024 * 1024


def dispatch(server: MediaServer, message: dict) -> dict | None:
    """Keep media results unchanged; negotiate the HTTP transport's MCP version."""
    if message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
        return {"jsonrpc": "2.0", "id": message.get("id"), "error": {"code": -32600, "message": "Invalid Request"}}
    response = server.handle(message)
    if response is not None and message["method"] == "initialize" and "result" in response:
        requested = (message.get("params") or {}).get("protocolVersion")
        response["result"]["protocolVersion"] = requested if requested in PROTOCOLS else "2025-06-18"
    return response


def make_handler(media: MediaServer, instance: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 30

        def log_message(self, *args: object) -> None:
            # RPC arguments and returned media never enter task setup logs.
            pass

        def reply(self, status: int, body: dict | None = None) -> None:
            # Count and validate before headers without retaining two full media-response copies.
            encoder = json.JSONEncoder(allow_nan=False)
            size = sum(len(piece.encode()) for piece in encoder.iterencode(body)) if body is not None else 0
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(size))
            self.send_header("Connection", "close")
            self.end_headers()
            if body is not None:
                for piece in encoder.iterencode(body):
                    self.wfile.write(piece.encode())
            self.close_connection = True

        def do_GET(self) -> None:
            if self.path == "/health":
                self.reply(200, {"service": "claweval-media", "instance": instance})
            else:
                # Stateless JSON responses: an SSE listener is deliberately unsupported.
                self.reply(405)

        def do_DELETE(self) -> None:
            self.reply(204 if self.path == "/mcp" else 404)

        def do_POST(self) -> None:
            if self.path != "/mcp":
                self.reply(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_REQUEST_BYTES:
                    self.reply(413)
                    return
                message = json.loads(self.rfile.read(size))
                if not isinstance(message, dict):
                    self.reply(400)
                    return
                result = dispatch(media, message)
                self.reply(202 if result is None else 200, result)
            except (ValueError, TypeError, UnicodeError):
                self.reply(400)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    args = parser.parse_args()
    if not args.instance or len(args.instance) > 128:
        parser.error("A bounded instance identity is required")
    # Single-threaded dispatch preserves the stdio tool's serialized frame budget.
    with HTTPServer((HOST, PORT), make_handler(MediaServer(), args.instance)) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
