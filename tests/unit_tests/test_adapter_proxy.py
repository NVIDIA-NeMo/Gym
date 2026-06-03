# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Integration tests for ``start_adapter_proxy``.

Drives a real uvicorn proxy thread against a stub upstream (a FastAPI app
hosted by Starlette's ``TestClient``-equivalent in-thread) and asserts:

  - adapted POST routes (``/v1/chat/completions``, ``/v1/messages``) run
    the pipeline (logging interceptor fires)
  - non-adapted paths (``/v1/models`` GET, ``/health``) pass through
    without running the pipeline
  - localhost-bind enforcement (``host="0.0.0.0"`` rejected)
  - user-supplied ``endpoint`` interceptor rejected
  - multi-Set-Cookie response headers survive the proxy
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, Response

from nemo_gym.adapters import start_adapter_proxy


# ---------------------------------------------------------------------------
# In-thread stub upstream — records requests, lets us inject responses
# ---------------------------------------------------------------------------


class _StubUpstream:
    def __init__(self) -> None:
        self.received: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.app = FastAPI()
        self._build_routes()
        self.port: int | None = None
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def _build_routes(self) -> None:
        @self.app.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
        )
        async def echo(path: str, request: Request) -> Response:
            try:
                body = json.loads(await request.body() or b"{}")
            except Exception:
                body = None
            with self._lock:
                self.received.append({"method": request.method, "path": "/" + path, "body": body})
            # Default: OpenAI-compat success
            if request.method == "POST":
                resp_body = {
                    "id": "stub-1",
                    "object": "chat.completion",
                    "model": body.get("model", "stub") if isinstance(body, dict) else "stub",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "stub-ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
                # Inject two Set-Cookie headers to exercise multi-cookie path
                out = JSONResponse(resp_body, status_code=200)
                out.raw_headers.append((b"set-cookie", b"a=1; Path=/"))
                out.raw_headers.append((b"set-cookie", b"b=2; HttpOnly"))
                return out
            return JSONResponse({"path": "/" + path, "method": request.method})

    def start(self) -> None:
        import socket

        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        cfg = uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="warning", access_log=False)
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        # Wait until ready
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/healthcheck", timeout=1) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError("stub upstream did not become healthy")

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@pytest.fixture
def upstream():
    s = _StubUpstream()
    s.start()
    try:
        yield s
    finally:
        s.stop()


def _post_json(url: str, body: dict) -> tuple[int, bytes, list[tuple[str, str]]]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read(), list(r.headers.items())
    except urllib.error.HTTPError as e:
        return e.code, e.read(), list(e.headers.items())


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_proxy_runs_pipeline_on_adapted_routes(upstream, caplog) -> None:
    proxy = start_adapter_proxy(
        upstream_url=upstream.url,
        adapters=[{"name": "logging", "config": {}}, {"name": "logging", "config": {}}],
    )
    try:
        with caplog.at_level(logging.INFO, logger="nemo_gym.adapters.interceptors.request_logging"):
            status, body, _ = _post_json(
                f"{proxy.url}/v1/chat/completions",
                {"model": "stub", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert status == 200, body
        assert json.loads(body)["choices"][0]["message"]["content"] == "stub-ok"
        assert any("request POST /v1/chat/completions" in r.message for r in caplog.records)
    finally:
        proxy.stop()


def test_proxy_passthrough_does_not_run_pipeline(upstream, caplog) -> None:
    proxy = start_adapter_proxy(
        upstream_url=upstream.url,
        adapters=[{"name": "logging", "config": {}}],
    )
    try:
        with caplog.at_level(logging.INFO, logger="nemo_gym.adapters.interceptors.request_logging"):
            status, _ = _get(f"{proxy.url}/v1/models")
        # Stub upstream returns 200 for GET /v1/models (via its catch-all)
        assert status == 200
        # Crucially, the logging interceptor should NOT have fired for the passthrough
        assert not any("request GET /v1/models" in r.message for r in caplog.records)
    finally:
        proxy.stop()


def test_proxy_multi_set_cookie_preserved(upstream) -> None:
    proxy = start_adapter_proxy(
        upstream_url=upstream.url,
        adapters=[{"name": "logging", "config": {}}],
    )
    try:
        status, _body, headers = _post_json(
            f"{proxy.url}/v1/chat/completions",
            {"model": "stub", "messages": []},
        )
        assert status == 200
        cookies = [v for k, v in headers if k.lower() == "set-cookie"]
        assert len(cookies) == 2, f"expected 2 Set-Cookie headers, got {len(cookies)}: {cookies}"
    finally:
        proxy.stop()


def test_proxy_rejects_remote_host_by_default() -> None:
    with pytest.raises(ValueError, match="refusing host"):
        start_adapter_proxy(upstream_url="http://x", adapters=[], host="0.0.0.0")


def test_proxy_allows_remote_with_explicit_flag(upstream) -> None:
    proxy = start_adapter_proxy(
        upstream_url=upstream.url,
        adapters=[],
        host="0.0.0.0",
        unsafe_allow_remote=True,
        port=0,
    )
    try:
        assert proxy.url.startswith("http://0.0.0.0:")
    finally:
        proxy.stop()


def test_proxy_rejects_user_supplied_endpoint() -> None:
    with pytest.raises(ValueError, match="endpoint.*cannot be used"):
        start_adapter_proxy(
            upstream_url="http://x",
            adapters=[{"name": "endpoint", "config": {"upstream_url": "http://y"}}],
        )


def test_proxy_handle_context_manager(upstream) -> None:
    with start_adapter_proxy(upstream_url=upstream.url, adapters=[]) as proxy:
        status, _ = _get(f"{proxy.url}/_proxy_health")
        assert status == 200
    # After context exit, the server should be stopped — port should reject
    # connections after a brief moment. We don't strictly assert this since
    # uvicorn shutdown is async, but the context manager should have called stop().
