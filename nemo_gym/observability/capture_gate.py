# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Ingress gate middleware for the model server.

The gate wraps the model server's app and, per request:

  1. Correlate the call to a rollout: strip a /ng-rollout/<rid> URL prefix, or
     read the x-nemo-gym-rollout-id header.
  2. Guard the namespace: under the /ng-rollout prefix, only the model
     endpoints are reachable (nothing else, especially not the capture reader).
  3. Forward the request and response through UNCHANGED, sniffing the request
     body and buffering a copy of the response as they flow. It never pre-reads
     and replays the request: a replayed receive that later returns
     http.disconnect trips a StreamingResponse's disconnect listener and aborts
     the stream. Forwarding unchanged is what lets streaming (SSE) responses
     pass through intact while still being captured.
  4. After the app finishes, record the call off the hot path: a ModelCallRecord
     always, and a TokenEntry when the response carries token-id fields and
     training capture is on. Non-streaming bodies are parsed directly; streaming
     bodies are reassembled from the buffered SSE.

Token ids are driven by the model server's own config (it returns them or not),
not by the gate rewriting the request, so the gate only ever observes.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Optional

from pydantic import BaseModel

from nemo_gym.observability.capture_store import LocalJsonlCaptureStore, WriteTracker
from nemo_gym.observability.records import AUX_TAG_HEADER, TOKEN_FIELDS, ModelCallRecord, TokenEntry
from nemo_gym.observability.token_sink import TokenSink, reset_token_sink, set_token_sink
from nemo_gym.rollout_id import NG_ROLLOUT_PREFIX_RE, ROLLOUT_ID_HEADER


MODEL_ENDPOINTS = ("/v1/responses", "/v1/chat/completions", "/v1/messages")


class ObservabilityConfig(BaseModel):
    enabled: bool = True
    capture_dir: str = "ng_captures"
    run_id: str = "run"
    fsync_per_record: bool = True
    # Training mode: record a TokenEntry when the served response carries
    # token-id fields. The model server produces those fields from its own
    # config; the gate never injects request flags (it only observes).
    return_token_ids: bool = False


def extract_token_fields(response_json: dict) -> Optional[dict]:
    """Pull the token-id fields (prompt_token_ids, generation_token_ids,
    generation_log_probs, ...) off a served response.

    Handles both response shapes the model server can return: Responses-style
    `output` items (the fields ride the last item that carries them) and
    chat-completions `choices[*].message`.
    """
    candidates: list[dict] = []
    for item in response_json.get("output") or []:
        if isinstance(item, dict) and item.get("generation_token_ids") is not None:
            candidates.append(item)
    for choice in response_json.get("choices") or []:
        msg = (choice or {}).get("message") or {}
        if isinstance(msg, dict) and msg.get("generation_token_ids") is not None:
            candidates.append(msg)
    if not candidates:
        return None
    src = candidates[-1]
    return {k: src.get(k) for k in TOKEN_FIELDS}


def strip_token_fields(response_json: dict) -> dict:
    def _strip(obj):
        if isinstance(obj, dict):
            return {k: _strip(v) for k, v in obj.items() if k not in TOKEN_FIELDS}
        if isinstance(obj, list):
            return [_strip(v) for v in obj]
        return obj

    return _strip(response_json)


def _strip_body_bytes(raw: bytes) -> bytes:
    try:
        obj = json.loads(raw or b"{}")
    except Exception:
        return raw
    return json.dumps(strip_token_fields(obj)).encode()


def _fold_cache_tokens(tokens_in, usage: dict) -> int:
    """Return the true prompt size in tokens.

    Anthropic /v1/messages reports input_tokens as the uncached remainder only,
    so cache-read + cache-creation tokens are folded back in. OpenAI / Responses
    usage already includes cached tokens in input_tokens (their nested
    cached_tokens is a subset), so those shapes are left untouched — the
    top-level cache_* keys are Anthropic-only, so this never double counts.
    """
    cache_read = usage.get("cache_read_input_tokens")
    cache_creation = usage.get("cache_creation_input_tokens")
    if cache_read is not None or cache_creation is not None:
        return int(tokens_in or 0) + int(cache_read or 0) + int(cache_creation or 0)
    return int(tokens_in or 0)


def _dialect_for(path: str) -> str:
    # Dialects are named after the API (open set keyed to the converter registry).
    if path.endswith("/messages"):
        return "messages"
    if path.endswith("/chat/completions"):
        return "chat_completions"
    return "responses"


def _cache_stats(usage: dict) -> Optional[dict]:
    """Normalize cache token counts across dialects (OpenAI nests
    cached_tokens under prompt_tokens_details; Anthropic uses top-level
    cache_read/creation). Returns None when the response reports no cache."""
    if not usage:
        return None
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens")
    cache_read = usage.get("cache_read_input_tokens")
    cache_creation = usage.get("cache_creation_input_tokens")
    out: dict = {}
    if cached is not None:
        out["cached_tokens"] = int(cached)
    if cache_read is not None:
        out["cache_read_tokens"] = int(cache_read)
    if cache_creation is not None:
        out["cache_creation_tokens"] = int(cache_creation)
    return out or None


class IngressGateMiddleware:
    """Pure-ASGI middleware (never BaseHTTPMiddleware, which buffers streams)."""

    def __init__(
        self,
        app,
        config: ObservabilityConfig,
        store: LocalJsonlCaptureStore | None = None,
        tracker: WriteTracker | None = None,
        model_name: str = "",
    ):
        self.app = app
        self.cfg = config
        self.store = store or LocalJsonlCaptureStore(
            config.capture_dir, run_id=config.run_id, fsync_per_record=config.fsync_per_record
        )
        self.tracker = tracker or WriteTracker()
        self.model_name = model_name

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        rollout_id: Optional[str] = headers.get(ROLLOUT_ID_HEADER)
        from_prefix = False

        m = NG_ROLLOUT_PREFIX_RE.match(path)
        if m:
            rollout_id, rest = m.group("rid"), m.group("rest")
            from_prefix = True
            # Namespace guard: the sandbox-facing prefix reaches model endpoints
            # and nothing else (especially not the capture read API).
            if not rest.startswith(MODEL_ENDPOINTS):
                await _plain_response(send, 404, b'{"error":"not found"}')
                return
            scope = dict(scope)
            scope["path"] = rest
            scope["raw_path"] = rest.encode()
            path = rest

        is_model_call = path.startswith(MODEL_ENDPOINTS) and scope["method"] == "POST"
        record_this = self.cfg.enabled and is_model_call and rollout_id is not None
        if not record_this:
            await self.app(scope, receive, send)
            return

        # Sniff the request body as the app reads it, and forward every response
        # message unchanged while buffering a copy. This preserves streaming and
        # avoids the false-disconnect that a pre-read + replayed receive causes.
        req_body = bytearray()

        async def sniff_receive():
            message = await receive()
            if message.get("type") == "http.request":
                req_body.extend(message.get("body", b"") or b"")
            return message

        started = time.monotonic()
        state: dict = {
            "status": 500,
            "headers": [],
            "streaming": False,
            "body": bytearray(),
            "ttft_ms": None,
            "held_start": None,
        }

        async def forward_send(message):
            mtype = message["type"]
            if mtype == "http.response.start":
                state["status"] = message["status"]
                state["headers"] = message.get("headers", [])
                ctype = next((v.decode() for k, v in state["headers"] if k.decode().lower() == "content-type"), "")
                state["streaming"] = "text/event-stream" in ctype
                # For a sandbox-facing (prefix) non-streaming response, hold the
                # start line so content-length can be rewritten after token
                # fields are stripped below. Everything else forwards straight
                # through, which keeps streaming (SSE) intact.
                if from_prefix and not state["streaming"]:
                    state["held_start"] = message
                    return
                await send(message)
                return
            if mtype == "http.response.body":
                chunk = message.get("body", b"") or b""
                if chunk and state["ttft_ms"] is None:
                    state["ttft_ms"] = (time.monotonic() - started) * 1000.0
                state["body"].extend(chunk)
                if from_prefix and not state["streaming"]:
                    # Buffer, then emit once complete with token fields removed:
                    # token ids never cross to the sandbox, in any mode.
                    if not message.get("more_body", False):
                        stripped = _strip_body_bytes(bytes(state["body"]))
                        start = state["held_start"] or {"type": "http.response.start", "status": state["status"]}
                        hdrs = [(k, v) for k, v in start.get("headers", []) if k.decode().lower() != "content-length"]
                        hdrs.append((b"content-length", str(len(stripped)).encode()))
                        await send({**start, "headers": hdrs})
                        await send({"type": "http.response.body", "body": stripped, "more_body": False})
                    return
                await send(message)
                return
            await send(message)

        # One request_id per model call, minted up front and shared: the gate
        # stamps it on the ModelCallRecord, and the streaming token sink stamps it on
        # the TokenEntry, so training can join them 1:1.
        request_id = uuid.uuid4().hex
        sink = TokenSink(
            rollout_id=rollout_id,
            request_id=request_id,
            store=self.store,
            tracker=self.tracker,
            model=self.model_name,
            enabled=self.cfg.return_token_ids,
        )
        sink_token = set_token_sink(sink)
        try:
            await self.app(scope, sniff_receive, forward_send)
        finally:
            reset_token_sink(sink_token)

        # Record off the hot path (parse + SSE reassembly + fsync'd append).
        self.tracker.track(
            rollout_id,
            asyncio.to_thread(
                self._record,
                rollout_id=rollout_id,
                request_id=request_id,
                from_prefix=from_prefix,
                aux_tag=headers.get(AUX_TAG_HEADER),
                path=path,
                req_body=bytes(req_body),
                status=int(state["status"] or 0),
                resp_body=bytes(state["body"]),
                streaming=bool(state["streaming"]),
                latency_ms=(time.monotonic() - started) * 1000.0,
                ttft_ms=state["ttft_ms"],
            ),
        )

    def _record(
        self,
        *,
        rollout_id: str,
        request_id: str,
        from_prefix: bool,
        aux_tag: Optional[str],
        path: str,
        req_body: bytes,
        status: int,
        resp_body: bytes,
        streaming: bool,
        latency_ms: float,
        ttft_ms: Optional[float],
    ) -> None:
        """Build and append the ModelCallRecord (+ TokenEntry) for one model call.

        Runs in a worker thread (append is a synchronous flock'd write). Fully
        guarded: a malformed body never surfaces as an error after the response
        already went out. Token ids are not present on a streamed wire, so for
        streaming calls the TokenEntry is recorded at the served layer (see
        token_sink); here it is recorded from the buffered non-streaming response.
        """
        dialect = _dialect_for(path)
        try:
            req_json = json.loads(req_body or b"{}")
        except Exception:
            req_json = {}
        if not isinstance(req_json, dict):
            req_json = {}

        token_info = None
        cache = None
        if streaming:
            response, tokens_in, tokens_out = _reassemble_sse(resp_body, dialect)
        else:
            try:
                resp_json = json.loads(resp_body or b"{}")
            except Exception:
                resp_json = {}
            if not isinstance(resp_json, dict):
                resp_json = {}
            token_info = extract_token_fields(resp_json)
            usage = resp_json.get("usage") or {}
            tokens_in = _fold_cache_tokens(usage.get("input_tokens") or usage.get("prompt_tokens") or 0, usage)
            tokens_out = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
            cache = _cache_stats(usage)
            response = strip_token_fields(resp_json)

        model = str(req_json.get("model") or self.model_name)
        self.store.append(
            ModelCallRecord(
                rollout_id=rollout_id,
                request_id=request_id,
                dialect=dialect,
                model=model,
                status_code=status,
                error_category=None if 0 < status < 400 else f"http_{status}",
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cache=cache,
                aux_tag=aux_tag,
                from_untrusted_prefix=from_prefix,
                request=req_json,
                response=response,
                streamed=streaming,
            )
        )
        if token_info is not None and self.cfg.return_token_ids:
            self.store.append(
                TokenEntry(
                    rollout_id=rollout_id,
                    request_id=request_id,
                    prompt_token_ids=token_info.get("prompt_token_ids") or [],
                    generation_token_ids=token_info.get("generation_token_ids") or [],
                    generation_log_probs=token_info.get("generation_log_probs") or [],
                    routed_experts=token_info.get("routed_experts"),
                    model=model,
                )
            )


def _reassemble_sse(raw: bytes, dialect: str) -> tuple[dict, int, int]:
    """Parse a raw SSE byte stream into (response_dict, tokens_in, tokens_out).
    Handles the Anthropic Messages event stream and the OpenAI chat.completion
    chunk stream. Fully guarded against non-dict / malformed events."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    tokens_in = tokens_out = 0
    for block in raw.split(b"\n\n"):
        data_lines = [ln[5:].strip() for ln in block.split(b"\n") if ln.startswith(b"data:")]
        for data in data_lines:
            if not data or data == b"[DONE]":
                continue
            try:
                ev = json.loads(data)
            except Exception:
                continue
            if not isinstance(ev, dict):
                continue
            etype = ev.get("type")
            if dialect == "messages":
                if etype == "message_start":
                    u = (ev.get("message") or {}).get("usage") or {}
                    tokens_in = _fold_cache_tokens(u.get("input_tokens") or 0, u)
                    tokens_out = int(u.get("output_tokens") or 0)
                elif etype == "content_block_delta":
                    d = ev.get("delta") or {}
                    if isinstance(d, dict) and d.get("type") == "text_delta":
                        text_parts.append(d.get("text") or "")
                elif etype == "content_block_start":
                    b = ev.get("content_block") or {}
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        tool_calls.append({"name": b.get("name"), "id": b.get("id")})
                elif etype == "message_delta":
                    u = ev.get("usage") or {}
                    if isinstance(u, dict) and u.get("output_tokens") is not None:
                        tokens_out = int(u["output_tokens"])
            else:  # OpenAI chat.completion chunks
                for ch in ev.get("choices") or []:
                    delta = (ch or {}).get("delta") or {}
                    if not isinstance(delta, dict):
                        continue
                    if delta.get("content"):
                        text_parts.append(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        fn = (tc or {}).get("function") or {}
                        tool_calls.append({"name": fn.get("name"), "id": tc.get("id")})
                usage = ev.get("usage") or {}
                if isinstance(usage, dict) and usage:
                    tokens_in = int(usage.get("prompt_tokens") or tokens_in)
                    tokens_out = int(usage.get("completion_tokens") or tokens_out)
    text = "".join(text_parts)
    if dialect == "messages":
        response = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "ng_tool_calls": tool_calls,
        }
    else:
        message: dict = {"role": "assistant", "content": text}
        if tool_calls:
            message["ng_tool_calls"] = tool_calls
        response = {"object": "chat.completion", "choices": [{"index": 0, "message": message}]}
    return response, tokens_in, tokens_out


async def _plain_response(send, status: int, body: bytes):
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


def install_ingress_gate(app, config: ObservabilityConfig, model_name: str = ""):
    """Attach the gate middleware and the capture read routes to a model
    server's FastAPI app. Call this once, after the app is built, from the
    model server's setup.
    """
    store = LocalJsonlCaptureStore(config.capture_dir, run_id=config.run_id, fsync_per_record=config.fsync_per_record)
    tracker = WriteTracker()
    from nemo_gym.observability.capture_reader import build_capture_router

    app.include_router(build_capture_router(store))
    app.add_middleware(_Adapter, config=config, store=store, tracker=tracker, model_name=model_name)
    return store, tracker


class _Adapter:
    """FastAPI's add_middleware wraps ASGI callables with kwargs."""

    def __init__(self, app, config, store, tracker, model_name=""):
        self._mw = IngressGateMiddleware(app, config, store, tracker, model_name)

    async def __call__(self, scope, receive, send):
        await self._mw(scope, receive, send)
