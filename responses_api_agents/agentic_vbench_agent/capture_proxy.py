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
  so the proxy records every served assistant message verbatim. The recorded
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
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from nemo_gym.chat_streaming import sanitize_streaming_chat_body, synthesize_chat_completion_sse
from nemo_gym.responses_converter import ResponsesConverter


AGENT_HEADER = "x-opencode-agent"
SESSION_HEADER = "x-opencode-session"
PARENT_SESSION_HEADER = "x-opencode-parent-session"
SPAWN_CALL_HEADER = "x-opencode-spawn-call"
TASK_CALL_HEADER = "x-opencode-task-call"
TASK_BINDING_HEADER = "x-opencode-task-binding"
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


def _recorded_completion(completion: dict[str, Any]) -> dict[str, Any]:
    """Keep routed-expert digests in diagnostics; native capture owns the routes.

    A single encoded route tensor can exceed 100 MB. Repeating it in every
    diagnostic file also makes episode finalization load gigabytes of unused
    data. Copy only the affected choices so forwarding and native token capture
    still receive the original completion, including its complete route tensor.
    """
    choices = []
    for choice in completion.get("choices") or []:
        routes = choice.get("routed_experts")
        if isinstance(routes, str):
            choice = {key: value for key, value in choice.items() if key != "routed_experts"}
            choice["routed_experts_summary"] = {
                "encoded_chars": len(routes),
                "sha256_utf8": hashlib.sha256(routes.encode()).hexdigest(),
            }
        choices.append(choice)
    return {**completion, "choices": choices}


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
    require_native_session_headers: bool = False
    require_delegation: bool = False
    turn: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Native subagent runs install an observation-only OpenCode headers hook.
    # Legacy callers without it are keyed by their leading request context.
    # In that legacy mode the first session is the
    # parent; a session whose opening user turn repeats the ``prompt`` argument of
    # a ``task`` tool call served earlier in this episode is that call's child.
    sessions: dict[str, SessionTrace] = field(default_factory=dict)
    task_prompts: dict[str, tuple[str, str]] = field(default_factory=dict)
    task_call_owners: dict[str, str] = field(default_factory=dict)
    task_bindings: dict[str, dict] = field(default_factory=dict)
    bound_turns: set[int] = field(default_factory=set)
    # A compaction request changes the system prompt, and its continuation may
    # replace the opening user turn. Those contexts still belong to the session
    # whose served history they replay; a changed prefix is not a Task spawn.
    message_owners: dict[str, set[str]] = field(default_factory=dict)
    compaction_messages: set[str] = field(default_factory=set)


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
        self,
        episode: str,
        upstream: str,
        record_dir: Path,
        *,
        local_step_cap_reply: bool = False,
        require_native_session_headers: bool = False,
        require_delegation: bool = False,
    ) -> str:
        """Route ``/avb/<episode>/...`` to ``upstream`` (a root URL without ``/v1``).

        ``local_step_cap_reply`` answers OpenCode's step-cap turn here instead of
        serving it: that request replays the history plus an assistant message
        OpenCode wrote itself, which ordinary (non framework-owned) token capture
        cannot attach to a served completion, so the ledger would poison the call
        and mask the whole rollout. The reply is never training data (the agent
        drops it as an ignored side call) and the verifier scores files, not text.

        ``require_delegation`` constrains the parent policy call to OpenCode's
        native ``task`` tool until a child session has actually reached inference.
        The upstream policy therefore generates the delegated call itself; no
        tool call or training token is synthesized by the environment.
        """
        if episode in self._routes:
            raise ValueError(f"Episode already registered: {episode}")
        record_dir.mkdir(parents=True, exist_ok=True)
        self._routes[episode] = EpisodeRoute(
            upstream=upstream.rstrip("/"),
            record_dir=record_dir,
            local_step_cap_reply=local_step_cap_reply,
            require_native_session_headers=require_native_session_headers,
            require_delegation=require_delegation,
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
        if request.method == "POST" and tail == "v1/session-bindings":
            return await self._bind_session(route, request)
        return await self._forward(route, request, tail)

    async def _bind_session(self, route: EpisodeRoute, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            fields = ("session_id", "parent_session_id", "delegation_call_id")
            if not isinstance(body, dict) or set(body) != set(fields):
                raise ValueError("Task binding requires native child, parent, and invocation IDs")
            if any(
                not isinstance(body[key], str) or not body[key] or body[key] != body[key].strip() for key in fields
            ):
                raise ValueError("Task binding requires nonempty native identifiers")
            # No await after parsing: a Task reports only after its own HTTP calls
            # finish. Other children may still be generating under route.lock;
            # their turns must neither delay this receipt nor be claimed by it.
            binding = _bind_native_task(route, **body)
        except (ValueError, TypeError) as error:
            errors = route.record_dir / "session_bindings" / "errors"
            errors.mkdir(parents=True, exist_ok=True)
            (errors / f"{uuid.uuid4().hex}.json").write_text(json.dumps({"error": str(error)}))
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response(binding)

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
            if request.headers.get(SESSION_HEADER):
                try:
                    session = observe_native_session(route, request.headers)
                except ValueError as error:
                    raise web.HTTPBadRequest(text=str(error)) from error
            elif route.require_native_session_headers:
                raise web.HTTPBadRequest(text="OpenCode subagent capture requires native session headers")
            else:
                session = observe_session(route, body, agent=agent)
            delegation_forced = False
            if (
                route.require_delegation
                and session.role == "parent"
                and not any(trace.role == "child" for trace in route.sessions.values())
            ):
                try:
                    force_task_tool_choice(body)
                except ValueError as error:
                    raise web.HTTPBadRequest(text=str(error)) from error
                delegation_forced = True
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
                **({"delegation_forced": True} if delegation_forced else {}),
                **({"native_task_binding_pending": True} if request.headers.get(TASK_BINDING_HEADER) else {}),
                **(
                    {"delegation_call_id": request.headers[TASK_CALL_HEADER]}
                    if TASK_CALL_HEADER in request.headers
                    else {}
                ),
                "streamed": streamed,
                "status": status,
                "elapsed_seconds": round(time.time() - started, 3),
                "request": {**body, "messages": redact_images(body.get("messages"))},
            }
            if status == 200:
                completion = json.loads(payload)
                record["response"] = _recorded_completion(completion)
                session.turns.append(turn)
                if isinstance(completion.get("id"), str):
                    session.response_ids.append(completion["id"])
                for call_id, prompt in task_tool_prompts(completion):
                    route.task_prompts[prompt.strip()] = (session.session_id, call_id)
                for call_id in task_tool_call_ids(completion):
                    route.task_call_owners[call_id] = session.session_id
                _remember_session_messages(route, session, completion, agent=agent)
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


def force_task_tool_choice(body: dict[str, Any]) -> None:
    """Require a policy-generated native OpenCode Task call for this turn."""
    tools = body.get("tools")
    if not isinstance(tools, list) or not any(
        isinstance(tool, dict)
        and tool.get("type") == "function"
        and isinstance(tool.get("function"), dict)
        and tool["function"].get("name") == "task"
        for tool in tools
    ):
        raise ValueError("Strict subagent mode requires OpenCode's native task tool in the policy request")
    body["tool_choice"] = {"type": "function", "function": {"name": "task"}}


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


def _first_user_text(body: dict, *, include_attachments: bool = True) -> str:
    """Read the opening turn, or its original OpenCode prompt text part."""
    for message in body.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if not include_attachments and isinstance(content, list):
                # OpenCode appends Read results for file references as separate
                # text parts. Only the original prompt identifies a Task spawn;
                # attachment text must never claim another Task's ownership.
                return next(
                    (part["text"] for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)),
                    "",
                )
            return _message_text(message)
    return ""


def _stable_session_environment(match: re.Match[str]) -> str:
    """Exclude OpenCode's changing environment date from session identity."""
    return re.sub(r"(?m)^([ \t]*Today's date:)[^\r\n]*", r"\1 <date>", match.group())


def session_key(body: dict) -> str:
    """Identify an OpenCode session by its leading context.

    OpenCode replays the full history on every call. Its environment date can
    change at UTC midnight, so normalize that generated field for this hash.
    The forwarded system prompt remains intact. The remaining system context
    and opening user turn distinguish the parent and each child's task prompt.
    """
    system = "\n".join(
        _message_text(m) for m in body.get("messages") or [] if isinstance(m, dict) and m.get("role") == "system"
    )
    system = re.sub(r"<env>.*?</env>", _stable_session_environment, system, flags=re.DOTALL)
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


def task_tool_call_ids(completion: dict) -> list[str]:
    """Return Task invocation IDs independently of their prompt or arguments."""
    return [
        call["id"]
        for choice in completion.get("choices") or []
        for call in (choice.get("message") or {}).get("tool_calls") or []
        if isinstance((call.get("function") or {}).get("name"), str)
        and call["function"]["name"].lower() == "task"
        and isinstance(call.get("id"), str)
        and call["id"]
    ]


def observe_native_session(route: EpisodeRoute, headers: Mapping[str, str]) -> SessionTrace:
    """Bind a native session to real, already-served Task calls without prompt inference."""
    session_id = headers.get(SESSION_HEADER)
    parent_id = headers.get(PARENT_SESSION_HEADER)
    spawn_id = headers.get(SPAWN_CALL_HEADER)
    call_id = headers.get(TASK_CALL_HEADER)
    deferred = headers.get(TASK_BINDING_HEADER) == "deferred"
    if headers.get(TASK_BINDING_HEADER) and not deferred:
        raise ValueError("Unknown native Task binding protocol")
    if not session_id or any(
        value is not None and (not value or value != value.strip())
        for value in (session_id, parent_id, spawn_id, call_id)
    ):
        raise ValueError("Native OpenCode session headers require nonempty identifiers")
    if parent_id:
        parent = route.sessions.get(parent_id)
        if parent is None or parent.role != "parent":
            raise ValueError("Native child session must belong to this episode's main session")
        if deferred:
            if spawn_id or call_id:
                raise ValueError("Deferred native Task headers cannot claim an unverified invocation")
        elif (
            not spawn_id
            or not call_id
            or any(route.task_call_owners.get(value) != parent_id for value in (spawn_id, call_id))
        ):
            raise ValueError("Native child session requires its served spawning and current Task calls")
        if any(
            spawn_id and trace.spawn_call_id == spawn_id and trace.session_id != session_id
            for trace in route.sessions.values()
        ):
            raise ValueError("A spawning Task call cannot own two native child sessions")
        role = "child"
    else:
        if (
            spawn_id
            or call_id
            or deferred
            or any(trace.role == "parent" and trace.session_id != session_id for trace in route.sessions.values())
        ):
            raise ValueError("Native main session has inconsistent ancestry")
        role = "parent"
    trace = route.sessions.get(session_id)
    if trace is not None:
        if (trace.role, trace.parent_session_id) != (role, parent_id) or (
            not deferred and trace.spawn_call_id != spawn_id
        ):
            raise ValueError("Native session identity changed role, parent, or original spawn")
        return trace
    trace = SessionTrace(session_id=session_id, role=role, parent_session_id=parent_id, spawn_call_id=spawn_id)
    route.sessions[session_id] = trace
    return trace


def _bind_native_task(
    route: EpisodeRoute, *, session_id: str, parent_session_id: str, delegation_call_id: str
) -> dict:
    child = route.sessions.get(session_id)
    if child is None or child.role != "child" or child.parent_session_id != parent_session_id:
        raise ValueError("Task binding requires an observed native child of this episode's parent")
    if route.task_call_owners.get(delegation_call_id) != parent_session_id:
        raise ValueError("Task binding does not belong to a served parent Task call")
    previous = route.task_bindings.get(delegation_call_id)
    if previous is not None:
        if previous["session_id"] != session_id:
            raise ValueError("A Task invocation cannot own two native children")
        return previous
    spawn_id = child.spawn_call_id or delegation_call_id
    turns = [turn for turn in child.turns if turn not in route.bound_turns]
    response_ids = [child.response_ids[child.turns.index(turn)] for turn in turns]
    binding = {
        "version": 1,
        "session_id": session_id,
        "parent_session_id": parent_session_id,
        "spawn_call_id": spawn_id,
        "delegation_call_id": delegation_call_id,
        "turns": turns,
        "response_ids": response_ids,
    }
    folder = route.record_dir / "session_bindings"
    folder.mkdir(exist_ok=True)
    path = folder / f"{hashlib.sha256(delegation_call_id.encode()).hexdigest()}.json"
    # Receipts are append-only. Raw model requests, completions, and ledger data
    # remain untouched; readers attach the verified ownership on copies.
    with path.open("x") as stream:
        stream.write(json.dumps(binding))
    child.spawn_call_id = spawn_id
    route.bound_turns.update(turns)
    route.task_bindings[delegation_call_id] = binding
    return binding


def _message_fingerprints(message: dict) -> list[str]:
    if message.get("role") != "assistant":
        return []
    fingerprints = []
    text = _message_text(message)
    if text.strip():
        fingerprints.append("text:" + hashlib.sha256(text.encode()).hexdigest())
    for call in message.get("tool_calls") or []:
        call_id = call.get("id")
        if isinstance(call_id, str) and call_id:
            fingerprints.append("call:" + call_id)
    return fingerprints


def _remember_session_messages(
    route: EpisodeRoute, session: SessionTrace, completion: dict, *, agent: str | None
) -> None:
    if agent in IGNORED_AGENTS:
        return
    for choice in completion.get("choices") or []:
        for fingerprint in _message_fingerprints(choice.get("message") or {}):
            route.message_owners.setdefault(fingerprint, set()).add(session.session_id)
            if agent == COMPACTION_AGENT:
                route.compaction_messages.add(fingerprint)


def _session_from_history(route: EpisodeRoute, body: dict, *, compaction_only: bool) -> SessionTrace | None:
    for message in reversed(body.get("messages") or []):
        if not isinstance(message, dict):
            continue
        owners = set()
        for fingerprint in _message_fingerprints(message):
            if compaction_only and fingerprint not in route.compaction_messages:
                continue
            candidates = route.message_owners.get(fingerprint, set())
            # Repeated generic replies (or equal summaries) cannot identify a
            # session. A unique tool-call id can still disambiguate that turn.
            if len(candidates) == 1:
                owners.update(candidates)
        if len(owners) == 1:
            owner = next(iter(owners))
            return next(trace for trace in route.sessions.values() if trace.session_id == owner)
    return None


def observe_session(route: EpisodeRoute, body: dict, *, agent: str | None = None) -> SessionTrace:
    """Resolve Task ownership and preserve it across OpenCode compaction.

    Compaction replays a session's served assistant messages. Its returned
    summary then identifies that same session when OpenCode rewrites the leading
    context. Only unambiguous served history is used; requests and completions
    are never changed. New Task prompts retain their explicit spawn identity.
    """
    key = session_key(body)
    spawn = route.task_prompts.get(_first_user_text(body, include_attachments=False).strip())
    is_compaction = (agent or classify_agent(None, body)) == COMPACTION_AGENT
    trace = None
    if is_compaction:
        trace = _session_from_history(route, body, compaction_only=False)
    elif not spawn:
        trace = _session_from_history(route, body, compaction_only=True)
    if trace is None and spawn:
        parent_id, call_id = spawn
        trace = next(
            (s for s in route.sessions.values() if s.parent_session_id == parent_id and s.spawn_call_id == call_id),
            None,
        )
        if trace is None:
            trace = SessionTrace(session_id=key, role="child", parent_session_id=parent_id, spawn_call_id=call_id)
    if trace is None:
        trace = route.sessions.get(key)
    if trace is None:
        if not route.sessions:
            trace = SessionTrace(session_id=key, role="parent")
        else:
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
        if record.get("native_task_binding_pending"):
            raise ValueError("Native child capture still has unresolved Task ownership")
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
        delegation_call_id = record.get("delegation_call_id")
        if delegation_call_id:
            calls = trace.setdefault("delegation_call_ids", [])
            if delegation_call_id not in calls:
                calls.append(delegation_call_id)
    return list(traces.values())


def load_records(record_dir: Path) -> list[dict]:
    records = sorted((json.loads(p.read_text()) for p in record_dir.glob("*.json")), key=lambda r: r["turn"])
    if not records:
        raise RuntimeError(f"No model calls were recorded under {record_dir}; inspect runner.log")
    _join_native_task_bindings(record_dir, records)
    return records


def _join_native_task_bindings(record_dir: Path, records: list[dict]) -> None:
    folder = record_dir / "session_bindings"
    if any((folder / "errors").glob("*.json")):
        raise ValueError("Native Task provenance failed; inspect session_bindings/errors before training")
    by_turn = {record["turn"]: record for record in records}
    owners = {
        call_id: record.get("session_id")
        for record in records
        if record.get("status", 200) == 200
        for call_id in task_tool_call_ids(record.get("response") or {})
    }
    births: dict[str, str] = {}
    invocations: dict[str, str] = {}
    claimed: set[int] = set()
    for path in sorted(folder.glob("*.json")):
        binding = json.loads(path.read_text())
        child, parent = binding["session_id"], binding["parent_session_id"]
        spawn, call = binding["spawn_call_id"], binding["delegation_call_id"]
        if binding["version"] != 1 or any(owners.get(value) != parent for value in (spawn, call)):
            raise ValueError("Native Task receipt does not match served parent calls")
        if births.setdefault(child, spawn) != spawn or invocations.setdefault(call, child) != child:
            raise ValueError("Native Task receipt changed a child birth or invocation owner")
        if len(binding["turns"]) != len(binding["response_ids"]):
            raise ValueError("Native Task receipt has inconsistent response ownership")
        for turn, response_id in zip(binding["turns"], binding["response_ids"], strict=True):
            record = by_turn.get(turn)
            if (
                turn in claimed
                or record is None
                or not record.get("native_task_binding_pending")
                or record.get("session_role") != "child"
                or record.get("session_id") != child
                or record.get("parent_session_id") != parent
                or (record.get("response") or {}).get("id") != response_id
            ):
                raise ValueError("Native Task receipt does not match exactly one recorded child response")
            claimed.add(turn)
            record.update(spawn_call_id=spawn, delegation_call_id=call, native_task_binding_pending=False)
    if any(invocations.get(spawn) != child for child, spawn in births.items()):
        raise ValueError("Native child capture is missing its original spawning Task receipt")
    if any(record.get("native_task_binding_pending") for record in records):
        raise ValueError("Native child capture still has unresolved Task ownership")


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
