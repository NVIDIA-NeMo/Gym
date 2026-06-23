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
"""Localhost adapter proxy.

Hosts the same ``AdapterPipeline`` as ``install_middleware`` in its own
uvicorn server. Used by agents whose SDK clients respect a ``*_BASE_URL``
env var (Anthropic / OpenAI / Cohere): the agent's outbound traffic is
pointed at the proxy URL, the proxy applies the chain, the proxy's own
upstream-call closure forwards to the real upstream.

Non-chat paths (``/v1/models``, batches, health checks) pass through to
the upstream verbatim so SDK pre-flight works.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, Response

from nemo_gym.adapters.pipeline import AdapterPipeline
from nemo_gym.adapters.types import AdapterRequest, AdapterResponse, GracefulError, InterceptorContext


logger = logging.getLogger(__name__)

# Routes whose POST traffic runs through the adapter pipeline. Everything
# else (any method, any path) is forwarded upstream verbatim so SDK
# pre-flight (model listing, batches, health checks) still works.
_ADAPTED_ROUTES = frozenset(
    {
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/messages",
        "/v1/embeddings",
    }
)


@dataclass
class ProxyHandle:
    """Handle for a running adapter proxy."""

    url: str
    port: int
    _server: uvicorn.Server
    _thread: threading.Thread

    def stop(self, timeout: float = 5.0) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=timeout)

    def __enter__(self) -> "ProxyHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def start_adapter_proxy(
    upstream_url: str,
    adapters: list[dict[str, Any]],
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    request_timeout: float = 120.0,
    health_timeout: float = 10.0,
    unsafe_allow_remote: bool = False,
) -> ProxyHandle:
    """Launch a localhost uvicorn that hosts the adapter pipeline.

    The proxy forwards adapted requests to ``upstream_url`` via its own
    ``aiohttp.ClientSession`` (created inside the proxy thread's event loop).
    Forwarding happens outside the user's interceptor chain via ``pipeline.
    process(req, upstream_call=...)`` — symmetric to ``install_middleware``
    using FastAPI's ``call_next``.
    """
    if host not in ("127.0.0.1", "localhost") and not unsafe_allow_remote:
        raise ValueError(
            f"start_adapter_proxy: refusing host={host!r} — the proxy forwards the "
            "client's Authorization header upstream, so binding to a non-localhost "
            "interface leaks credentials. Pass unsafe_allow_remote=True to override."
        )

    if any(s.get("name") == "endpoint" for s in adapters):
        raise ValueError(
            "start_adapter_proxy: the 'endpoint' interceptor cannot be used here — "
            "the proxy performs upstream forwarding itself. Drop it from `adapters`."
        )

    pipeline = AdapterPipeline.from_config(list(adapters))

    app = _build_app(pipeline, upstream_url.rstrip("/"), request_timeout)
    actual_port = _bind_port(host, port)

    config = uvicorn.Config(
        app,
        host=host,
        port=actual_port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    url = f"http://{host}:{actual_port}"
    _wait_for_health(url, timeout=health_timeout)
    logger.info("adapter proxy ready upstream=%s url=%s", upstream_url, url)

    return ProxyHandle(url=url, port=actual_port, _server=server, _thread=thread)


def _bind_port(host: str, preferred: int) -> int:
    if preferred:
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _wait_for_health(url: str, timeout: float) -> None:
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/_proxy_health", timeout=1) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError(f"adapter proxy at {url} did not become healthy in {timeout:.1f}s")


def _build_app(pipeline: AdapterPipeline, upstream_url: str, request_timeout: float) -> FastAPI:
    @asynccontextmanager
    async def _lifespan(app_: FastAPI):
        app_.state.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=request_timeout))
        try:
            yield
        finally:
            await app_.state.session.close()

    app = FastAPI(lifespan=_lifespan)
    app.state.pipeline = pipeline
    app.state.upstream_url = upstream_url
    app.state.request_timeout = request_timeout

    @app.get("/_proxy_health")
    async def _health() -> JSONResponse:
        return JSONResponse({"ok": True})

    async def _dispatch(path: str, request: Request) -> Response:
        norm_path = "/" + path.lstrip("/")
        if request.method == "POST" and norm_path in _ADAPTED_ROUTES:
            return await _run_pipeline(request, norm_path)
        return await _passthrough(request, norm_path)

    app.add_api_route(
        "/{path:path}",
        _dispatch,
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )

    return app


async def _run_pipeline(request: Request, path: str) -> Response:
    pipeline: AdapterPipeline = request.app.state.pipeline
    session: aiohttp.ClientSession = request.app.state.session
    upstream_url: str = request.app.state.upstream_url

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}},
            status_code=400,
        )

    adapter_req = AdapterRequest(
        method=request.method,
        path=path,
        headers=dict(request.headers),
        body=body,
        ctx=InterceptorContext(),
    )

    async def _upstream(req: AdapterRequest) -> AdapterResponse:
        target = f"{upstream_url}{req.path}"
        if request.url.query:
            target = f"{target}?{request.url.query}"
        fwd_headers = {k: v for k, v in req.headers.items() if k.lower() not in ("host", "content-length")}
        import json as _json

        t0 = time.perf_counter()
        async with session.post(target, data=_json.dumps(req.body), headers=fwd_headers) as resp:
            raw = await resp.read()
            latency = (time.perf_counter() - t0) * 1000
            # Preserve multi-valued response headers (Set-Cookie etc.)
            resp_headers: list[tuple[bytes, bytes]] = [
                (k.encode("latin-1"), v.encode("latin-1")) for k, v in resp.headers.items()
            ]
            try:
                parsed: Any = _json.loads(raw) if raw else {}
            except (ValueError, UnicodeDecodeError):
                parsed = raw

            return AdapterResponse(
                status_code=resp.status,
                headers=resp_headers,
                body=parsed,
                latency_ms=latency,
                ctx=req.ctx,
            )

    try:
        resp = await pipeline.process(adapter_req, upstream_call=_upstream)
    except GracefulError as exc:
        # Mirror middleware mode: budget/control-flow termination -> 429 (not 500).
        return JSONResponse(
            {
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "code": "session_budget_exhausted",
                },
            },
            status_code=429,
        )
    except Exception:
        logger.exception("adapter proxy pipeline error")
        return JSONResponse(
            {"error": {"message": "Internal adapter proxy error", "type": "server_error"}},
            status_code=500,
        )

    fwd_headers: list[tuple[bytes, bytes]] = []
    if isinstance(resp.headers, list):
        for name, value in resp.headers:
            lname = name.decode("latin-1").lower()
            if lname in ("content-length", "transfer-encoding", "content-encoding"):
                continue
            fwd_headers.append((name, value))
    elif isinstance(resp.headers, dict):
        for k, v in resp.headers.items():
            if k.lower() in ("content-length", "transfer-encoding", "content-encoding"):
                continue
            fwd_headers.append((k.encode("latin-1"), v.encode("latin-1")))

    if isinstance(resp.body, bytes):
        out: Response = Response(content=resp.body, status_code=resp.status_code)
    else:
        out = JSONResponse(content=resp.body, status_code=resp.status_code)

    fwd_keys_lower = {k.lower() for k, _ in fwd_headers}
    framing_kept = [
        (k, v) for k, v in out.raw_headers if k.lower() == b"content-length" or k.lower() not in fwd_keys_lower
    ]
    out.raw_headers = framing_kept + fwd_headers
    return out


async def _passthrough(request: Request, path: str) -> Response:
    """Forward request to upstream verbatim — for SDK pre-flight paths."""
    session: aiohttp.ClientSession = request.app.state.session
    upstream_url: str = request.app.state.upstream_url
    timeout: float = request.app.state.request_timeout

    body = await request.body()
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}

    target = f"{upstream_url}{path}"
    if request.url.query:
        target = f"{target}?{request.url.query}"

    try:
        async with session.request(
            method=request.method,
            url=target,
            data=body if body else None,
            headers=fwd_headers,
        ) as resp:
            raw = await resp.read()
            out_headers: list[tuple[bytes, bytes]] = []
            for k, v in resp.headers.items():
                if k.lower() in ("content-length", "transfer-encoding", "content-encoding"):
                    continue
                out_headers.append((k.encode("latin-1"), v.encode("latin-1")))
            out = Response(content=raw, status_code=resp.status)
            out.raw_headers = [(hk, hv) for hk, hv in out.raw_headers if hk.lower() == b"content-length"] + out_headers
            return out
    except asyncio.TimeoutError:
        return JSONResponse(
            {"error": {"message": f"Upstream timed out after {timeout}s", "type": "timeout"}},
            status_code=504,
        )
    except aiohttp.ClientError as exc:
        logger.warning("passthrough %s failed: %s", target, exc)
        return JSONResponse(
            {"error": {"message": f"Upstream error: {exc}", "type": "upstream_error"}},
            status_code=502,
        )


__all__ = ["start_adapter_proxy", "ProxyHandle"]
