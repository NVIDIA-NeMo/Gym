# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-episode recorder between OpenCode and Gym's rollout-scoped policy route.

Stock OpenCode runs inside the Harbor task container and talks to an
OpenAI-compatible endpoint with streamed Chat completions. NeMo-RL token
capture needs two things that endpoint cannot give it directly:

* framework-owned context (compaction training) accepts exactly one
  non-streaming Chat completion per call, so the proxy forwards OpenCode's
  request without ``stream`` and replays the complete completion as the SSE
  chunk stream OpenCode expects;
* NeMo-RL matches the agent's returned output to captured calls by content,
  so the proxy records every served completion verbatim. The recorded
  completions are converted with the same Chat conversion Gym's capture
  ledger applies, which makes the fingerprints identical by construction.

OpenCode tags each call with ``x-opencode-agent``; compaction summaries arrive
as ``compaction`` and are reported separately. The proxy never alters the
upstream decision: Gym's rollout prefix on the upstream URL attributes the
calls, and the vLLM worker stages the exact tokens.
"""

import asyncio
import hashlib
import json
import re
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from nemo_gym.chat_streaming import sanitize_streaming_chat_body, synthesize_chat_completion_sse
from nemo_gym.responses_converter import ResponsesConverter


AGENT_HEADER = "x-opencode-agent"
COMPACTION_AGENT = "compaction"
# Side calls whose generations are not part of the task trajectory. Title
# generation is disabled in the OpenCode config for training, but a stray call
# must still not be presented as accepted history.
IGNORED_AGENTS = frozenset({"title", "summary"})
_IMAGE_DATA_URL = re.compile(r"^data:(image/[^;]+);base64,(.*)$", re.DOTALL)
_FORWARDED_HEADER_PREFIXES = ("x-opencode-",)


def redact_images(messages: Any) -> Any:
    """Replace inline image payloads by their digest so records stay small."""
    if not isinstance(messages, list):
        return messages
    redacted = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            redacted.append(message)
            continue
        parts = []
        for part in message["content"]:
            url = part.get("image_url") if isinstance(part, dict) else None
            url = url.get("url") if isinstance(url, dict) else url
            match = _IMAGE_DATA_URL.match(url) if isinstance(url, str) else None
            if match:
                digest = hashlib.sha256(match.group(2).encode()).hexdigest()
                part = {**part, "image_url": {"url": f"data:{match.group(1)};sha256,{digest}"}}
            parts.append(part)
        redacted.append({**message, "content": parts})
    return redacted


def local_ip_toward(host: str, port: int = 80) -> str:
    """The local interface address that routes to ``host`` (no traffic is sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect((host, port))
        return probe.getsockname()[0]


@dataclass
class EpisodeRoute:
    upstream: str
    record_dir: Path
    turn: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class CaptureProxy:
    def __init__(self, *, advertise_host: str, port: int = 0, request_timeout_s: float | None = None) -> None:
        self.advertise_host = advertise_host
        self.port = port
        self.request_timeout_s = request_timeout_s
        self._routes: dict[str, EpisodeRoute] = {}
        self._runner: web.AppRunner | None = None
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        app = web.Application(client_max_size=1024**3)
        app.router.add_route("*", "/avb/{episode}/{tail:.*}", self._dispatch)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await site.start()
        self.port = self._runner.addresses[0][1]
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=self.request_timeout_s)
        )

    async def stop(self) -> None:
        if self._session is not None:
            await self._session.close()
        if self._runner is not None:
            await self._runner.cleanup()

    def register(self, episode: str, upstream: str, record_dir: Path) -> str:
        """Route ``/avb/<episode>/...`` to ``upstream`` (a root URL without ``/v1``)."""
        if episode in self._routes:
            raise ValueError(f"Episode already registered: {episode}")
        record_dir.mkdir(parents=True, exist_ok=True)
        self._routes[episode] = EpisodeRoute(upstream=upstream.rstrip("/"), record_dir=record_dir)
        return f"http://{self.advertise_host}:{self.port}/avb/{episode}/v1"

    def unregister(self, episode: str) -> None:
        self._routes.pop(episode, None)

    async def _dispatch(self, request: web.Request) -> web.StreamResponse:
        route = self._routes.get(request.match_info["episode"])
        if route is None:
            raise web.HTTPNotFound(text="unknown episode")
        tail = request.match_info["tail"]
        if request.method == "POST" and tail == "v1/chat/completions":
            return await self._chat(route, request)
        return await self._forward(route, request, tail)

    def _upstream_headers(self, request: web.Request) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        for name, value in request.headers.items():
            if name.lower().startswith(_FORWARDED_HEADER_PREFIXES):
                headers[name] = value
        return headers

    async def _forward(self, route: EpisodeRoute, request: web.Request, tail: str) -> web.Response:
        assert self._session is not None
        body = await request.read()
        async with self._session.request(
            request.method,
            f"{route.upstream}/{tail}",
            data=body or None,
            headers=self._upstream_headers(request) if body else None,
            params=request.query,
        ) as upstream:
            payload = await upstream.read()
            return web.Response(status=upstream.status, body=payload, content_type=upstream.content_type)

    async def _chat(self, route: EpisodeRoute, request: web.Request) -> web.StreamResponse:
        assert self._session is not None
        body = await request.json()
        streamed = body.get("stream") is True
        include_usage = False
        if streamed:
            body, include_usage = sanitize_streaming_chat_body(body)
        agent = request.headers.get(AGENT_HEADER)
        started = time.time()
        # OpenCode is sequential within a session; the lock also keeps a stray
        # overlapping side call from racing the ledger's sequential admission.
        async with route.lock:
            turn = route.turn
            route.turn += 1
            async with self._session.post(
                f"{route.upstream}/v1/chat/completions", json=body, headers=self._upstream_headers(request)
            ) as upstream:
                status = upstream.status
                payload = await upstream.read()
                content_type = upstream.content_type
            record: dict[str, Any] = {
                "turn": turn,
                "agent": agent,
                "session": request.headers.get("x-opencode-session"),
                "streamed": streamed,
                "status": status,
                "elapsed_seconds": round(time.time() - started, 3),
                "request": {**body, "messages": redact_images(body.get("messages"))},
            }
            if status == 200:
                completion = json.loads(payload)
                record["response"] = completion
            else:
                record["error"] = payload.decode(errors="replace")[:4000]
            (route.record_dir / f"{turn:04d}.json").write_text(json.dumps(record))
        if status != 200:
            return web.Response(status=status, body=payload, content_type=content_type)
        if streamed:
            events = "".join(synthesize_chat_completion_sse(completion, include_usage=include_usage))
            return web.Response(
                body=events.encode(), content_type="text/event-stream", headers={"Cache-Control": "no-cache"}
            )
        return web.Response(body=payload, content_type="application/json")


def load_records(record_dir: Path) -> list[dict]:
    records = sorted((json.loads(p.read_text()) for p in record_dir.glob("*.json")), key=lambda r: r["turn"])
    if not records:
        raise RuntimeError(f"No model calls were recorded under {record_dir}; inspect runner.log")
    return records


@dataclass
class CaptureStats:
    model_calls: int = 0
    compaction_summaries: int = 0
    ignored_calls: int = 0
    failed_calls: int = 0
    generated_tokens: int = 0
    terminal_response_id: str | None = None
    complete: bool = False


def capture_output(
    records: list[dict], *, include_compaction_summaries: bool = True
) -> tuple[list[dict], CaptureStats]:
    """Rebuild the served assistant output in the ledger's own item shape.

    Mirrors ``nemo_gym.token_id_capture.completion.completion_items``: the
    gateway already applied its reasoning conversion, so Chat text is not
    re-split and reasoning fields are not echoed. Ids on message items are
    fresh; function_call ids are the served tool-call ids, which NeMo-RL uses
    as item identities.
    """
    converter = ResponsesConverter(return_token_id_information=False, uses_reasoning_parser=False)
    items: list[dict] = []
    stats = CaptureStats(model_calls=len(records))
    last: dict | None = None
    for record in records:
        if record.get("status", 200) != 200 or "response" not in record:
            stats.failed_calls += 1
            continue
        agent = record.get("agent")
        if agent in IGNORED_AGENTS:
            stats.ignored_calls += 1
            continue
        if agent == COMPACTION_AGENT:
            stats.compaction_summaries += 1
            if not include_compaction_summaries:
                continue
        response = record["response"]
        choices = response.get("choices") or []
        if len(choices) != 1:
            raise ValueError(f"Expected exactly one choice in model call {record['turn']}")
        items.extend(
            item.model_dump(mode="json")
            for item in converter.postprocess_assistant_message_dict(choices[0]["message"])
        )
        stats.generated_tokens += (response.get("usage") or {}).get("completion_tokens") or 0
        last = record
    if last is not None:
        response = last["response"]
        terminal = response.get("id")
        if not isinstance(terminal, str) or not terminal:
            raise ValueError("Served response is missing its id; capture cannot identify the terminal call")
        stats.terminal_response_id = terminal
        choice = response["choices"][0]
        message = choice["message"]
        stats.complete = (
            choice.get("finish_reason") == "stop" and bool(message.get("content")) and not message.get("tool_calls")
        )
    return items, stats
