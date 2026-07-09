# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Per-rollout model-call capture for model servers.

Opt-in, off by default. A pure-ASGI middleware records every /v1/responses,
/v1/chat/completions, and /v1/messages exchange -- including failed calls -- into a
per-rollout CaptureStore, forwarding bytes downstream unchanged so it composes with
streaming (SSE) responses. Best-effort; never alters the response. Correlation is
OpenAI-compatible: callers set the x-nemo-gym-rollout-id header or use a
/ng-rollout/<rollout_id>/v1/... base_url prefix, which is stripped before routing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from nemo_gym.model_call_capture import CaptureStore, aggregate_model_call_records, read_model_call_records
from nemo_gym.server_utils import ROLLOUT_HEADER, ROLLOUT_PATH_PREFIX, rollout_id_from_run_body


logger = logging.getLogger(__name__)

_OBSERVED_PATHS = {
    "/v1/responses": "responses",
    "/v1/chat/completions": "chat",
    "/v1/messages": "messages",
}


def _scope_header(scope: dict[str, Any], name: str) -> Optional[str]:
    """Read a request header (case-insensitive) from a raw ASGI scope."""
    target = name.lower().encode("latin-1")
    for key, value in scope.get("headers") or []:
        if key.lower() == target:
            return value.decode("latin-1")
    return None


def _headers_content_type(headers: list) -> bytes:
    for key, value in headers:
        if key.lower() == b"content-type":
            return value
    return b""


# Consumer side of the URL-prefix protocol: strip /ng-rollout/<id> before routing, key capture by
# <id>. The constant + producer (apply_rollout_prefix) are in server_utils.
_ROLLOUT_PATH_RE = re.compile(rf"^/{re.escape(ROLLOUT_PATH_PREFIX)}/(?P<rollout_id>[^/]+)(?P<rest>/.*)$")


# --- Model-call exchange capture ---
def _default_capture_dir(server_name: str) -> str:
    env_dir = os.environ.get("NEMO_GYM_MODEL_CALL_CAPTURE_DIR")
    if env_dir:
        return env_dir
    return str(Path(tempfile.gettempdir()) / "nemo_gym_model_calls" / server_name)


def make_capture_store(config: Any) -> Optional[CaptureStore]:
    """Build a CaptureStore when observability is enabled; otherwise None."""
    if not getattr(config, "observability_enabled", False):
        return None
    root = getattr(config, "model_call_capture_dir", None) or _default_capture_dir(
        getattr(config, "name", None) or "model_server"
    )
    try:
        return CaptureStore(root)
    except Exception:
        logger.warning("Could not initialize model-call capture at %s; disabling it.", root, exc_info=True)
        return None


def _classify_status(status_code: int) -> Optional[str]:
    """Normalized error_category from an HTTP status (None when < 400)."""
    if status_code < 400:
        return None
    if status_code in (408, 504):
        return "timeout"
    if status_code == 429:
        return "rate_limit"
    if status_code in (401, 403):
        return "auth"
    if status_code == 404:
        return "not_found"
    if status_code < 500:
        return "client_error"
    return "upstream_error"


def _classify_exception(exc: BaseException) -> str:
    """Normalized error_category for an exception raised while calling the model."""
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "timeout"
    if "conn" in name:
        return "connection"
    return "exception"


# --- SSE reconstruction: rebuild a final response object from a streamed body ---
def _parse_sse_events(raw: bytes) -> list[dict[str, Any]]:
    """Parse an SSE byte stream into its JSON ``data:`` payloads (best-effort; non-JSON skipped)."""
    events: list[dict[str, Any]] = []
    for block in raw.decode("utf-8", errors="replace").split("\n\n"):
        data_lines = [line[5:].lstrip() for line in block.splitlines() if line.startswith("data:")]
        if not data_lines:
            continue
        payload = "\n".join(data_lines)
        if payload == "[DONE]":
            continue
        try:
            parsed = json.loads(payload)
        except Exception:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def _reconstruct_anthropic_sse(events: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Rebuild a complete Anthropic Messages response from its streamed events."""
    message: Optional[dict[str, Any]] = None
    blocks: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    tool_json: dict[int, str] = {}
    for event in events:
        etype = event.get("type")
        if etype == "message_start":
            msg = event.get("message") or {}
            message = {k: msg.get(k) for k in ("id", "type", "role", "model", "stop_reason") if msg.get(k) is not None}
            usage.update(msg.get("usage") or {})
        elif etype == "content_block_start":
            blocks[event.get("index", len(blocks))] = dict(event.get("content_block") or {})
        elif etype == "content_block_delta":
            idx = event.get("index", 0)
            block = blocks.setdefault(idx, {})
            delta = event.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                block["type"] = block.get("type") or "text"
                block["text"] = (block.get("text") or "") + (delta.get("text") or "")
            elif dtype == "thinking_delta":
                block["type"] = block.get("type") or "thinking"
                block["thinking"] = (block.get("thinking") or "") + (delta.get("thinking") or "")
            elif dtype == "input_json_delta":
                tool_json[idx] = tool_json.get(idx, "") + (delta.get("partial_json") or "")
        elif etype == "message_delta":
            usage.update(event.get("usage") or {})
            stop = (event.get("delta") or {}).get("stop_reason")
            if message is not None and stop:
                message["stop_reason"] = stop
    if message is None and not blocks:
        return None
    content = []
    for idx in sorted(blocks):
        block = blocks[idx]
        if block.get("type") == "tool_use" and idx in tool_json and not block.get("input"):
            try:
                block["input"] = json.loads(tool_json[idx]) if tool_json[idx] else {}
            except Exception:
                block["input"] = {"_raw": tool_json[idx]}
        content.append(block)
    result: dict[str, Any] = {**(message or {}), "type": "message", "content": content}
    if usage:
        result["usage"] = usage
    return result


def _reconstruct_chat_sse(events: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Rebuild a Chat Completions response from streamed chunks."""
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    usage: Optional[dict[str, Any]] = None
    model: Optional[str] = None
    role = "assistant"
    finish_reason: Optional[str] = None
    saw_choice = False
    for chunk in events:
        model = chunk.get("model") or model
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            saw_choice = True
            delta = choice.get("delta") or {}
            role = delta.get("role") or role
            if delta.get("content"):
                content_parts.append(delta["content"])
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                reasoning_parts.append(reasoning)
            for tc in delta.get("tool_calls") or []:
                slot = tool_calls.setdefault(
                    tc.get("index", 0), {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    if not saw_choice:
        return None
    message: dict[str, Any] = {"role": role, "content": "".join(content_parts) or None}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    result: dict[str, Any] = {
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if usage:
        result["usage"] = usage
    return result


def _reconstruct_responses_sse(events: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Rebuild a Responses API response: the terminal envelope carries the full response object."""
    for event in reversed(events):
        if event.get("type") in ("response.completed", "response.incomplete", "response.failed") and isinstance(
            event.get("response"), dict
        ):
            return event["response"]
    for event in reversed(events):
        if isinstance(event.get("response"), dict):
            return event["response"]
    return None


def _reconstruct_streamed_response(raw: bytes, dialect: str) -> Optional[dict[str, Any]]:
    """Best-effort: reassemble a final response object from a streamed (SSE) body, by dialect."""
    events = _parse_sse_events(raw)
    if not events:
        return None
    if dialect == "messages":
        return _reconstruct_anthropic_sse(events)
    if dialect == "responses":
        return _reconstruct_responses_sse(events)
    return _reconstruct_chat_sse(events)


def _record(
    store: CaptureStore,
    dialect: str,
    config: Any,
    request_bytes: bytes,
    *,
    rollout_id: str,
    response_body: Any,
    status_code: Optional[int],
    error_category: Optional[str],
    latency_ms: float,
    ttft_ms: Optional[float] = None,
) -> None:
    """Append one exchange (success or failure). Best-effort: never raises."""
    try:
        store.record(
            rollout_id,
            {
                "dialect": dialect,
                "model_server": getattr(config, "name", None),
                "latency_ms": round(latency_ms, 2),
                "latency_ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
                "status_code": status_code,
                "error_category": error_category,
                "request": json.loads(request_bytes) if request_bytes else None,
                "response": response_body,
            },
        )
    except Exception:
        logger.warning("Model-call capture failed for one %s call.", dialect, exc_info=True)


class _CaptureMiddleware:
    """Pure-ASGI per-rollout capture.

    Always strips an optional ``/ng-rollout/<id>`` path prefix before routing (used as the capture
    key; an explicit header wins) so the prefix is a stable routing feature independent of capture.
    When ``store`` is set it buffers the request body and a copy of the response while forwarding both
    downstream unchanged, so it composes with streaming (SSE) responses -- it never consumes or rewraps
    the stream. Each SSE chunk is forwarded immediately and also appended to an in-memory buffer for
    post-hoc reassembly, so a very long stream is held in memory until it completes. When ``store`` is
    None (capture disabled) it strips the prefix and forwards only.
    """

    def __init__(self, app: Any, *, store: Optional[CaptureStore], config: Any) -> None:
        self._app = app
        self._store = store
        self._config = config

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        path = scope.get("path", "")
        rollout_from_path: Optional[str] = None
        prefix_match = _ROLLOUT_PATH_RE.match(path)
        if prefix_match:
            rollout_from_path = prefix_match.group("rollout_id")
            path = prefix_match.group("rest")
            scope = {**scope, "path": path, "raw_path": path.encode("utf-8")}

        # Capture disabled: the prefix is already stripped (routing preserved), so just forward.
        if self._store is None:
            await self._app(scope, receive, send)
            return

        dialect = _OBSERVED_PATHS.get(path)
        if dialect is None:
            await self._app(scope, receive, send)  # not observed (or a stripped non-/v1 path)
            return

        rollout_id = _scope_header(scope, ROLLOUT_HEADER) or rollout_from_path or "rollout"
        request_body = bytearray()

        async def _receive() -> dict[str, Any]:
            message = await receive()
            if message.get("type") == "http.request":
                request_body.extend(message.get("body", b"") or b"")
            return message

        state: dict[str, Any] = {"status": None, "streaming": False, "body": bytearray(), "ttft_ms": None}
        start = time.perf_counter()

        async def _send(message: dict[str, Any]) -> None:
            message_type = message.get("type")
            if message_type == "http.response.start":
                state["status"] = message.get("status")
                content_type = _headers_content_type(message.get("headers") or [])
                state["streaming"] = content_type.startswith(b"text/event-stream")
            elif message_type == "http.response.body":
                chunk = message.get("body", b"") or b""
                if chunk and state["ttft_ms"] is None:
                    state["ttft_ms"] = (time.perf_counter() - start) * 1000.0
                state["body"].extend(chunk)  # buffered for both shapes; SSE is reassembled below
            await send(message)  # forward unchanged -> streaming is preserved

        try:
            await self._app(scope, _receive, _send)
        except Exception as exc:
            # Offload the blocking write+fsync so it never stalls the event loop.
            await asyncio.to_thread(
                _record,
                self._store,
                dialect,
                self._config,
                bytes(request_body),
                rollout_id=rollout_id,
                response_body=None,
                status_code=None,
                error_category=_classify_exception(exc),
                latency_ms=(time.perf_counter() - start) * 1000.0,
                ttft_ms=state["ttft_ms"],
            )
            raise

        latency_ms = (time.perf_counter() - start) * 1000.0
        status = state["status"]
        body_bytes = bytes(state["body"])
        streaming = state["streaming"]
        ttft_ms = state["ttft_ms"]
        request_bytes = bytes(request_body)
        store, config = self._store, self._config

        def _parse_and_record() -> None:
            # Off the event loop: body parse + SSE reassembly is best-effort and fully guarded, so a
            # malformed body can never surface as an ASGI error after the response was already sent.
            #
            # Ordering: the response is forwarded before this fsynced write runs, so a call becomes
            # durable slightly after its bytes reach the agent. The rollout merge re-reads the capture
            # JSONL, and the agent -> orchestrator /run round-trip
            # that precedes any merge dominates this sub-fsync window, so the final call is present in
            # practice; num_calls is always recomputed from the durable file.
            response_body = None
            if body_bytes:
                try:
                    response_body = (
                        _reconstruct_streamed_response(body_bytes, dialect) if streaming else json.loads(body_bytes)
                    )
                except Exception:
                    response_body = None
            error_category = _classify_status(status) if status is not None else None
            # A 2xx whose body we couldn't parse/reassemble isn't a clean success -- flag it so it
            # doesn't silently count as a success with null tokens in reliability/cost sums.
            if error_category is None and body_bytes and response_body is None:
                error_category = "capture_parse_error"
            _record(
                store,
                dialect,
                config,
                request_bytes,
                rollout_id=rollout_id,
                response_body=response_body,
                status_code=status,
                error_category=error_category,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
            )

        await asyncio.to_thread(_parse_and_record)


def install_model_call_capture(app: Any, config: Any) -> None:
    """Install model-call capture middleware.

    Always installed so the ``/ng-rollout/<id>`` correlation prefix is stripped before routing
    regardless of whether capture is enabled (otherwise a default ``gym eval`` would 404 on every
    prefixed model call). When capture is enabled the middleware additionally records each observed
    call's request + response into a rollout-keyed CaptureStore while forwarding bytes downstream
    unchanged (streamed SSE bodies are forwarded as they arrive and also buffered for reassembly).
    """
    app.add_middleware(_CaptureMiddleware, store=make_capture_store(config), config=config)


# --- Consumer read: fold per-rollout capture into the rollout record (uniform across agents) ---
_model_call_capture_dirs_warned = False


def _warn_model_call_capture_dirs_unresolved() -> None:
    """Warn once when capture is enabled but no readable capture dir was resolved (silent no-op guard)."""
    global _model_call_capture_dirs_warned
    if _model_call_capture_dirs_warned:
        return
    _model_call_capture_dirs_warned = True
    logger.warning(
        "Model-call capture is enabled but no capture directory was resolved to merge from -- "
        "per-rollout calls will not be attached. Set NEMO_GYM_MODEL_CALL_CAPTURE_DIR (shared by the "
        "model server and rollout collection) for a deterministic location."
    )


def model_call_capture_dirs_from_config(global_config_dict: Any, env: Optional[dict[str, str]] = None) -> list[Path]:
    """Candidate directories to read per-rollout captures from.

    Collects ``$NEMO_GYM_MODEL_CALL_CAPTURE_DIR`` plus the resolved directory of every
    observability-enabled model server in the global config. Deduped; existing dirs only.
    """
    environ = env if env is not None else os.environ
    dirs: list[Path] = []
    shared = environ.get("NEMO_GYM_MODEL_CALL_CAPTURE_DIR")
    if shared:
        dirs.append(Path(shared))

    # Normalize an OmegaConf config to plain containers so the walk below sees dict/list nodes.
    try:
        from omegaconf import DictConfig as _DictConfig
        from omegaconf import OmegaConf

        if isinstance(global_config_dict, _DictConfig):
            global_config_dict = OmegaConf.to_container(global_config_dict, resolve=False)
    except Exception:
        pass

    saw_enabled = False

    def _walk(node: Any, key: Optional[str] = None, instance_key: Optional[str] = None) -> None:
        nonlocal saw_enabled
        if isinstance(node, dict):
            if node.get("observability_enabled"):
                saw_enabled = True
                # The producer (make_capture_store) keys its default dir off ``config.name`` -- the
                # top-level server-instance key (== server_name), not the leaf impl key the node sits
                # under. Resolve off that instance key so the consumer reads the producer's directory.
                resolved = (
                    node.get("model_call_capture_dir")
                    or shared
                    or _default_capture_dir(instance_key or node.get("name") or key or "model_server")
                )
                dirs.append(Path(resolved))
            for child_key, value in node.items():
                _walk(value, child_key, instance_key if instance_key is not None else child_key)
        elif isinstance(node, (list, tuple)):
            for value in node:
                _walk(value, key, instance_key)

    try:
        _walk(global_config_dict)
    except Exception:
        logger.debug("Could not resolve capture dirs from the global config.", exc_info=True)

    if not saw_enabled:
        return []

    seen: set[Path] = set()
    unique: list[Path] = []
    for directory in dirs:
        if directory not in seen and directory.exists():
            seen.add(directory)
            unique.append(directory)
    if saw_enabled and not unique:
        _warn_model_call_capture_dirs_unresolved()
    return unique


def _store_for_rollout(rollout_id: str, capture_dirs: list[Path]) -> Optional[CaptureStore]:
    for directory in capture_dirs:
        store = CaptureStore(directory)
        if store.path_for(rollout_id).exists():
            return store
    return None


def clear_model_call_captures_for_rollouts(records: list[Any], capture_dirs: list[Path]) -> None:
    """Remove stale per-rollout capture files for these records before a fresh (non-resume) run.

    Capture files are keyed by a deterministic rollout id (task-rollout-attempt), so without this a
    re-run would append onto the previous run's capture for the same id. Best-effort; run-scopes the
    capture so each run's model-call evidence stays isolated.
    """
    if not capture_dirs:
        return
    for directory in capture_dirs:
        try:
            store = CaptureStore(directory)
        except Exception:
            continue
        for record in records:
            rollout_id = rollout_id_from_run_body(record)
            if rollout_id:
                store.path_for(rollout_id).unlink(missing_ok=True)


def merge_model_call_capture_into_record(
    record: dict[str, Any], capture_dirs: list[Path], *, include_payloads: bool = False
) -> dict[str, Any]:
    """Attach captured model-call observability data to a rollout record in place.

    Keyed by the rollout id derived from the record's task/rollout/attempt indices, so the attached
    shape is identical for every agent harness. Adds
    ``ng_model_call_capture = {rollout_id, metrics, calls}`` where ``calls`` are derived observability
    records. Raw request and response payloads remain in the capture store unless ``include_payloads``
    is true. No-op when no capture exists. The harness output and reward are not modified.
    """
    if not capture_dirs:
        return record
    rollout_id = rollout_id_from_run_body(record)
    if rollout_id is None:
        return record
    store = _store_for_rollout(rollout_id, capture_dirs)
    if store is None:
        return record
    calls = read_model_call_records(store, rollout_id)
    if not calls:
        return record
    exclude = None if include_payloads else {"request", "response"}
    record["ng_model_call_capture"] = {
        "rollout_id": rollout_id,
        "metrics": aggregate_model_call_records(calls),
        "calls": [call.model_dump(exclude=exclude) for call in calls],
    }
    return record
