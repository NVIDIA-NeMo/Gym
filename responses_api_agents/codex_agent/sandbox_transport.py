# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox-local forwarding for Codex's disabled native HTTP 429 retries.

The task image only promises Python's standard library, not Gym or aiohttp.
Synchronous requests run in the forwarding thread and preserve streamed bodies.
"""

import json
import math
from contextlib import contextmanager
from http.client import HTTPException, IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from time import sleep
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MAX_RATE_LIMIT_RETRIES = 2


@contextmanager
def rate_limit_proxy(base_url: str, invocation_id: str, attempts_path: Path, *, timeout: float) -> Iterator[str]:
    """Apply the local adapter's two-retry policy inside the task sandbox."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            pass

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in {"host", "content-length", "connection", "x-session-id"}
            }
            headers["x-session-id"] = invocation_id
            for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
                try:
                    upstream = urlopen(Request(f"{base_url}/responses", data=body, headers=headers), timeout=timeout)
                except HTTPError as error:
                    upstream = error
                except (URLError, OSError) as error:
                    with attempts_path.open("a") as attempts:
                        attempts.write(json.dumps({"status": None, "error": str(error)}) + "\n")
                    self.send_error(502, "Upstream model connection failed")
                    return
                with upstream:
                    record = {"status": upstream.status}
                    try:
                        if upstream.status == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
                            upstream.read()
                            try:
                                delay = float(upstream.headers.get("Retry-After", 2**attempt))
                            except ValueError:
                                delay = 2**attempt
                            if not math.isfinite(delay):
                                delay = 2**attempt
                            sleep(min(max(delay, 0), 30))
                            continue
                        self.send_response(upstream.status)
                        for key, value in upstream.headers.items():
                            if key.lower() not in {"content-length", "transfer-encoding", "connection"}:
                                self.send_header(key, value)
                        self.send_header("Connection", "close")
                        self.end_headers()
                        while True:
                            try:
                                chunk = upstream.read1(64 * 1024)
                                if not chunk and upstream.length:
                                    raise IncompleteRead(b"", upstream.length)
                            except (OSError, HTTPException) as error:
                                record["transport_error"] = str(error)
                                break
                            if not chunk:
                                break
                            # A downstream write failure can be Codex cancellation, not a model failure.
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        self.close_connection = True
                        return
                    finally:
                        with attempts_path.open("a") as attempts:
                            attempts.write(json.dumps(record) + "\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
