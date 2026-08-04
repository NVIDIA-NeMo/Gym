# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Minimal streaming turn-counter proxy (PinchBench PoC).

Sits between an external harness (OpenClaw) and a streaming-capable model
endpoint. Counts each POST as one agent turn; injects threshold budget
reminders into the request ``messages``; rejects with 429 once ``max_turns``
is exceeded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal
from urllib.parse import urlparse

from aiohttp import ClientSession, web

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}

_WARN_THRESHOLD = 0.80
_URGENT_THRESHOLD = 0.95

Position = Literal["system_message", "user_message"]


class _Severity(str, Enum):
    URGENT = "urgent"
    WARN = "warn"
    NON_ACTIONABLE = "non_actionable"


def _threshold_severity(n: int, max_turns: int) -> _Severity:
    ratio = n / max_turns
    if ratio >= _URGENT_THRESHOLD:
        return _Severity.URGENT
    if ratio >= _WARN_THRESHOLD:
        return _Severity.WARN
    return _Severity.NON_ACTIONABLE


def _threshold_message_body(n: int, max_turns: int, remaining: int, severity: _Severity) -> str:
    if severity is _Severity.URGENT:
        return (
            f"URGENT: Turn {n}/{max_turns} — only {remaining} turn(s) left. "
            f"You MUST provide your final answer NOW. Do not start new work."
        )
    return (
        f"Turn {n}/{max_turns} — {remaining} turns remaining. "
        f"Begin wrapping up: finish current work and prepare your final answer."
    )


def _append_to_last_user_message(messages: list, notice: str) -> None:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            tail = content[-1] if content else None
            if isinstance(tail, dict) and tail.get("type") == "text":
                existing = tail.get("text", "")
                tail["text"] = f"{existing}\n\n{notice}" if existing else notice
            else:
                content.append({"type": "text", "text": notice})
        elif isinstance(content, str):
            msg["content"] = f"{content}\n\n{notice}" if content else notice
        else:
            msg["content"] = notice
        return


def inject_turn_reminder(
    body: dict[str, Any],
    *,
    n: int,
    max_turns: int,
    position: Position = "system_message",
) -> dict[str, Any]:
    """Mutate (and return) a chat-completions body with a threshold turn reminder when due."""
    if position not in ("system_message", "user_message"):
        raise ValueError(f"invalid position: {position!r}")

    messages = body.get("messages")
    if not isinstance(messages, list):
        return body

    severity = _threshold_severity(n, max_turns)
    if severity is _Severity.NON_ACTIONABLE:
        return body

    remaining = max_turns - n
    text = _threshold_message_body(n, max_turns, remaining, severity)
    notice = f"[SYSTEM] {text}" if position == "system_message" else text

    if position == "system_message":
        messages.append({"role": "system", "content": notice})
    else:
        _append_to_last_user_message(messages, notice)
    return body


@dataclass
class TurnCounterProxy:
    """Running proxy handle. ``base_url`` is what to put in ``MODEL_BASE_URL``."""

    base_url: str
    port: int
    _runner: web.AppRunner
    _site: web.TCPSite
    _session: ClientSession
    _state: dict

    @property
    def turns_used(self) -> int:
        return int(self._state["turns_used"])

    async def stop(self) -> None:
        await self._site.stop()
        await self._runner.cleanup()
        await self._session.close()


async def start_turn_counter_proxy(
    *,
    upstream_base_url: str,
    api_key: str,
    max_turns: int,
    position: Position = "system_message",
    host: str = "127.0.0.1",
    advertise_host: str | None = None,
) -> TurnCounterProxy:
    """Start a per-task proxy that enforces ``max_turns`` on POSTs.

    ``upstream_base_url`` is the real OpenAI-compatible base (e.g. ``http://host/v1``).
    Returned ``base_url`` mirrors that path prefix on the local listen address.
    """
    if max_turns < 1:
        raise ValueError(f"max_turns must be >= 1, got {max_turns}")
    inject_turn_reminder({"messages": []}, n=1, max_turns=max_turns, position=position)

    upstream = urlparse(upstream_base_url.rstrip("/"))
    if not upstream.scheme or not upstream.netloc:
        raise ValueError(f"invalid upstream_base_url: {upstream_base_url!r}")
    upstream_origin = f"{upstream.scheme}://{upstream.netloc}"
    path_prefix = upstream.path.rstrip("/")  # e.g. "/v1"

    state = {"turns_used": 0}
    session = ClientSession()

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "turns_used": state["turns_used"], "max_turns": max_turns})

    async def proxy_post(request: web.Request) -> web.StreamResponse:
        state["turns_used"] += 1
        n = state["turns_used"]
        if n > max_turns:
            return web.json_response(
                {
                    "error": {
                        "message": f"Turn budget exhausted: {n}/{max_turns} turns used.",
                        "type": "invalid_request_error",
                        "code": "session_budget_exhausted",
                    }
                },
                status=429,
            )

        forward_url = f"{upstream_origin}{request.path_qs}"
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
        if api_key and "authorization" not in {k.lower() for k in headers}:
            headers["Authorization"] = f"Bearer {api_key}"

        raw = await request.read()
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict):
            inject_turn_reminder(payload, n=n, max_turns=max_turns, position=position)
            body = json.dumps(payload).encode()
            headers = {k: v for k, v in headers.items() if k.lower() != "content-length"}
        else:
            body = raw

        upstream_resp = await session.request(
            request.method,
            forward_url,
            data=body,
            headers=headers,
            allow_redirects=False,
        )

        out_headers = {k: v for k, v in upstream_resp.headers.items() if k.lower() not in _HOP_BY_HOP}
        response = web.StreamResponse(status=upstream_resp.status, headers=out_headers)
        await response.prepare(request)
        async for chunk in upstream_resp.content.iter_any():
            await response.write(chunk)
        await response.write_eof()
        upstream_resp.release()
        return response

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_route("POST", "/{path:.*}", proxy_post)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, 0)
    await site.start()
    sockets = site._server.sockets  # noqa: SLF001 — aiohttp does not expose the bound port otherwise
    port = sockets[0].getsockname()[1]

    adv = advertise_host or host
    base_url = f"http://{adv}:{port}{path_prefix}"
    return TurnCounterProxy(
        base_url=base_url,
        port=port,
        _runner=runner,
        _site=site,
        _session=session,
        _state=state,
    )
