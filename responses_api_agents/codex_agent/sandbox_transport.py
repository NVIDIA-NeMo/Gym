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
# Adapter-chosen memory cap for helper evidence, not a Codex protocol limit.
# Larger responses still forward in full; only the buffered evidence is dropped.
_MAX_COMPACTION_RESPONSE_BYTES = 8 * 1024 * 1024


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    return (
        "".join(block["text"] for block in content if isinstance(block, dict) and isinstance(block.get("text"), str))
        if isinstance(content, list)
        else ""
    )


def _compaction_request(body: bytes) -> bool:
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError):
        return False
    client_metadata = payload.get("client_metadata") if isinstance(payload, dict) else None
    if not isinstance(client_metadata, dict):
        return False
    # Tested compaction formats and upstream sources are documented in observability.py.
    metadata = client_metadata.get("x-codex-turn-metadata")
    if not isinstance(metadata, str):
        return False
    try:
        metadata = json.loads(metadata)
    except ValueError:
        return False
    return isinstance(metadata, dict) and metadata.get("request_kind") == "compaction"


def _compaction_response(body: bytearray) -> dict[str, str] | None:
    """Read one completed helper's identity and text without changing the forwarded SSE."""
    responses = []
    # A final unterminated frame was never dispatched to the SSE consumer.
    for frame in body.replace(b"\r\n", b"\n").split(b"\n\n")[:-1]:
        data = b"\n".join(line[5:].lstrip(b" ") for line in frame.splitlines() if line.startswith(b"data:"))
        try:
            event = json.loads(data)
        except (ValueError, UnicodeError):
            continue
        if not isinstance(event, dict) or event.get("type") != "response.completed":
            continue
        response = event.get("response")
        if not isinstance(response, dict) or response.get("status") != "completed":
            return None
        response_id, output = response.get("id"), response.get("output")
        if not isinstance(response_id, str) or not response_id or not isinstance(output, list):
            return None
        summary = "".join(
            _message_text(item.get("content"))
            for item in output
            if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "assistant"
        )
        if not summary:
            return None
        responses.append({"response_id": response_id, "summary": summary})
    return responses[0] if len(responses) == 1 else None


@contextmanager
def rate_limit_proxy(base_url: str, invocation_id: str, attempts_path: Path, *, timeout: float) -> Iterator[str]:
    """Apply the local adapter's two-retry policy inside the task sandbox."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            pass

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            compacting = _compaction_request(body)
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
                    # Keep only bounded helper evidence. Policy response bodies are
                    # never buffered here; the shared model capture owns those.
                    response_body = bytearray() if compacting and upstream.status == 200 else None
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
                            if response_body is not None:
                                if len(response_body) + len(chunk) <= _MAX_COMPACTION_RESPONSE_BYTES:
                                    response_body.extend(chunk)
                                else:
                                    response_body = None
                            # A downstream write failure can be Codex cancellation, not a model failure.
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        self.close_connection = True
                        return
                    finally:
                        if response_body is not None and not record.get("transport_error"):
                            helper = _compaction_response(response_body)
                            if helper is not None:
                                record["compaction_response"] = helper
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
