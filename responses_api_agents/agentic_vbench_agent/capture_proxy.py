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

Calls are attributed to OpenCode agents so compaction summaries can be
reported (and optionally excluded) separately: an ``x-opencode-agent`` header
when OpenCode sends one, otherwise the request shape. Stock OpenCode 1.14.39
sends no such header; its compaction request is the only one without tools,
and its user turn is one of two fixed prompts (``session/compaction.ts``). The
proxy never alters the upstream decision: Gym's rollout prefix on the upstream
URL attributes the calls, and the vLLM worker stages the exact tokens.
"""

import asyncio
import hashlib
import json
import re
import socket
import time
import uuid
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
IGNORED_AGENTS = frozenset({"title", "summary", STEP_CAP_AGENT := "step_cap"})
# OpenCode 1.14.39 session/prompt.ts: once the agent's step budget is spent it
# appends this assistant message to the history and asks for a text-only reply.
STEP_CAP_PROMPT_PREFIX = "CRITICAL - MAXIMUM STEPS REACHED"
STEP_CAP_REPLY = "Maximum steps reached for this session; no further tool calls can be made."
# OpenCode 1.14.39 session/compaction.ts: the user turn of a compaction request.
COMPACTION_PROMPT_PREFIXES = (
    "Create a new anchored summary from the conversation history above.",
    "Update the anchored summary below using the conversation history above.",
)
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
class SessionTrace:
    """One OpenCode session (the parent, or a depth-one child) within an episode."""

    session_id: str
    role: str  # "parent" | "child"
    parent_session_id: str | None = None
    spawn_call_id: str | None = None
    response_ids: list = field(default_factory=list)
    turns: list = field(default_factory=list)


@dataclass
class EpisodeRoute:
    upstream: str
    record_dir: Path
    local_step_cap_reply: bool = False
    turn: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Stock OpenCode sends no session header; sessions are keyed by the leading
    # context of each request (see ``session_key``). The first session is the
    # parent; a session whose opening user turn repeats the ``prompt`` argument of
    # a ``task`` tool call served earlier in this episode is that call's child.
    sessions: dict = field(default_factory=dict)
    task_prompts: dict = field(default_factory=dict)  # prompt text -> (parent session id, tool call id)


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

    def register(
        self, episode: str, upstream: str, record_dir: Path, *, local_step_cap_reply: bool = False
    ) -> str:
        """Route ``/avb/<episode>/...`` to ``upstream`` (a root URL without ``/v1``).

        ``local_step_cap_reply`` answers OpenCode's step-cap turn here instead of
        serving it: that request replays the history plus an assistant message
        OpenCode wrote itself, which ordinary (non framework-owned) token capture
        cannot attach to a served completion, so the ledger would poison the call
        and mask the whole rollout. The reply is never training data (the agent
        drops it as an ignored side call) and the verifier scores files, not text.
        """
        if episode in self._routes:
            raise ValueError(f"Episode already registered: {episode}")
        record_dir.mkdir(parents=True, exist_ok=True)
        self._routes[episode] = EpisodeRoute(
            upstream=upstream.rstrip("/"), record_dir=record_dir, local_step_cap_reply=local_step_cap_reply
        )
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
        agent = classify_agent(request.headers.get(AGENT_HEADER), body)
        started = time.time()
        if route.local_step_cap_reply and is_step_cap_request(body):
            async with route.lock:
                turn = route.turn
                route.turn += 1
                completion = step_cap_completion(body.get("model"))
                record = {
                    "turn": turn,
                    "agent": STEP_CAP_AGENT,
                    "session": request.headers.get("x-opencode-session"),
                    "streamed": streamed,
                    "status": 200,
                    "local": True,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "request": {**body, "messages": redact_images(body.get("messages"))},
                    "response": completion,
                }
                (route.record_dir / f"{turn:04d}.json").write_text(json.dumps(record))
            if streamed:
                events = "".join(synthesize_chat_completion_sse(completion, include_usage=include_usage))
                return web.Response(
                    body=events.encode(), content_type="text/event-stream", headers={"Cache-Control": "no-cache"}
                )
            return web.Response(body=json.dumps(completion).encode(), content_type="application/json")
        # Calls are serialized per episode: Gym's capture ledger admits one call
        # at a time per rollout ("Cannot admit a call after incomplete capture"),
        # so parallel child sessions overlap on tool execution, not model calls.
        async with route.lock:
            turn = route.turn
            route.turn += 1
            session = observe_session(route, body)
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
                "session_id": session.session_id,
                "session_role": session.role,
                "parent_session_id": session.parent_session_id,
                "spawn_call_id": session.spawn_call_id,
                "streamed": streamed,
                "status": status,
                "elapsed_seconds": round(time.time() - started, 3),
                "request": {**body, "messages": redact_images(body.get("messages"))},
            }
            if status == 200:
                completion = json.loads(payload)
                record["response"] = completion
                session.turns.append(turn)
                if isinstance(completion.get("id"), str):
                    session.response_ids.append(completion["id"])
                for call_id, prompt in task_tool_prompts(completion):
                    route.task_prompts[prompt.strip()] = (session.session_id, call_id)
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


def _message_text(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def classify_agent(header_agent: str | None, body: dict) -> str | None:
    """Name the OpenCode agent behind a chat request.

    The header wins when present. Otherwise a request without tools whose last
    user turn is OpenCode's compaction prompt is the ``compaction`` agent; task
    turns always carry the tool set.
    """
    if header_agent:
        return header_agent
    if body.get("tools"):
        return None
    for message in reversed(body.get("messages") or []):
        if isinstance(message, dict) and message.get("role") == "user":
            text = _message_text(message).lstrip()
            if text.startswith(COMPACTION_PROMPT_PREFIXES):
                return COMPACTION_AGENT
            return None
    return None


def _first_user_text(body: dict) -> str:
    for message in body.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            return _message_text(message)
    return ""


def session_key(body: dict) -> str:
    """Identify an OpenCode session by its leading context.

    OpenCode replays the full history on every call, so the system prompt plus
    the opening user turn is stable within a session and differs between the
    parent (the benchmark prompt) and each child (its ``task`` prompt).
    """
    system = "\n".join(
        _message_text(m) for m in body.get("messages") or [] if isinstance(m, dict) and m.get("role") == "system"
    )
    digest = hashlib.sha256()
    digest.update(system.encode())
    digest.update(b"\x00")
    digest.update(_first_user_text(body).encode())
    return digest.hexdigest()[:16]


def task_tool_prompts(completion: dict) -> list[tuple[str, str]]:
    """``(tool_call_id, prompt)`` for every ``task`` tool call in a served completion."""
    prompts = []
    for choice in completion.get("choices") or []:
        for call in (choice.get("message") or {}).get("tool_calls") or []:
            function = call.get("function") or {}
            # OpenCode resolves tool names case-insensitively, so mirror that
            # behavior when attributing a spawned child to its task call.
            name = function.get("name")
            if not isinstance(name, str) or name.lower() != "task":
                continue
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except ValueError:
                continue
            prompt = arguments.get("prompt") if isinstance(arguments, dict) else None
            if isinstance(prompt, str) and prompt and isinstance(call.get("id"), str):
                prompts.append((call["id"], prompt))
    return prompts


def observe_session(route: EpisodeRoute, body: dict) -> SessionTrace:
    """Return the session a request belongs to, registering it on first sight."""
    key = session_key(body)
    trace = route.sessions.get(key)
    if trace is None:
        if not route.sessions:
            trace = SessionTrace(session_id=key, role="parent")
        else:
            spawn = route.task_prompts.get(_first_user_text(body).strip())
            parent_id, call_id = spawn if spawn else (next(iter(route.sessions)), None)
            trace = SessionTrace(session_id=key, role="child", parent_session_id=parent_id, spawn_call_id=call_id)
        route.sessions[key] = trace
    return trace


def is_step_cap_request(body: dict) -> bool:
    """True for OpenCode's text-only turn after its step budget is spent."""
    messages = body.get("messages") or []
    last = messages[-1] if messages else None
    return (
        isinstance(last, dict)
        and last.get("role") == "assistant"
        and not last.get("tool_calls")
        and _message_text(last).lstrip().startswith(STEP_CAP_PROMPT_PREFIX)
    )


def step_cap_completion(model: str | None) -> dict:
    """A minimal Chat completion answering the step-cap turn without the policy."""
    return {
        "id": f"local-step-cap-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or "policy",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": STEP_CAP_REPLY},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def session_traces(records: list[dict]) -> list[dict]:
    """Collapse per-call session tags into one ownership entry per session.

    Entries carry the served response ids each session owns (parent first, then
    children in order of first appearance) so the trainer can attribute capture
    chains to sessions. Records without session tags (older proxies) yield [].
    """
    traces: dict[str, dict] = {}
    for record in records:
        if record.get("status", 200) != 200 or "response" not in record or record.get("local"):
            continue
        session_id = record.get("session_id")
        if not session_id:
            continue
        trace = traces.setdefault(
            session_id,
            {
                "session_id": session_id,
                "session_role": record.get("session_role"),
                "parent_session_id": record.get("parent_session_id"),
                "spawn_call_id": record.get("spawn_call_id"),
                "response_ids": [],
            },
        )
        response_id = record["response"].get("id")
        if isinstance(response_id, str):
            trace["response_ids"].append(response_id)
    return list(traces.values())


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
    # Per-session ownership of the served calls (parent first), for subagent training.
    session_traces: list = field(default_factory=list)


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
    stats.session_traces = session_traces(records)
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
