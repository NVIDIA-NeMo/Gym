# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

## NOTE: hacky thing for preclusters like prenyx should not usually be needed 

from __future__ import annotations

import base64
import json
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from uuid import uuid4


PENDING: queue.Queue[str] = queue.Queue()
REQUESTS: dict[str, dict[str, Any]] = {}
LOCK = threading.Lock()
EXCLUDED_HEADERS = {"connection", "content-length", "host", "transfer-encoding"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"model-bridge {fmt % args}", flush=True)

    def _read(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length)

    def _send(
        self,
        status: int,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        response_headers = headers or {"content-type": "application/json"}
        for key, value in response_headers.items():
            if key.lower() not in EXCLUDED_HEADERS:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)
            self.wfile.flush()

    def _queue_request(self, method: str) -> None:
        request_id = uuid4().hex
        event = threading.Event()
        item = {
            "method": method,
            "path": self.path,
            "headers": {key: value for key, value in self.headers.items() if key.lower() not in EXCLUDED_HEADERS},
            "body": self._read(),
            "event": event,
            "reply": None,
        }
        with LOCK:
            REQUESTS[request_id] = item
        PENDING.put(request_id)
        if not event.wait(timeout=1800):
            with LOCK:
                REQUESTS.pop(request_id, None)
            self._send(504, b'{"error":"model bridge timeout"}')
            return
        with LOCK:
            reply = REQUESTS.pop(request_id)["reply"]
        self._send(
            int(reply.get("status", 200)),
            base64.b64decode(reply.get("body_b64") or ""),
            {str(key): str(value) for key, value in (reply.get("headers") or {}).items()},
        )

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, b"ready\n", {"content-type": "text/plain"})
            return
        if self.path == "/bridge/next":
            try:
                request_id = PENDING.get(timeout=1.0)
            except queue.Empty:
                self._send(204)
                return
            with LOCK:
                item = REQUESTS[request_id]
            payload = {
                "request_id": request_id,
                "method": item["method"],
                "path": item["path"],
                "headers": item["headers"],
                "body_b64": base64.b64encode(item["body"]).decode(),
            }
            self._send(200, json.dumps(payload).encode())
            return
        self._queue_request("GET")

    def do_POST(self) -> None:  # noqa: N802
        if self.path.startswith("/bridge/reply/"):
            request_id = self.path.rsplit("/", 1)[-1]
            reply = json.loads(self._read())
            with LOCK:
                item = REQUESTS.get(request_id)
                if item is None:
                    self._send(404, b'{"error":"unknown request"}')
                    return
                item["reply"] = reply
                item["event"].set()
            self._send(200, b'{"ok":true}')
            return
        self._queue_request("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._queue_request("DELETE")

    def do_PATCH(self) -> None:  # noqa: N802
        self._queue_request("PATCH")

    def do_PUT(self) -> None:  # noqa: N802
        self._queue_request("PUT")


def main() -> None:
    port = int(os.environ.get("BBH_MODEL_BRIDGE_PORT", "18080"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
