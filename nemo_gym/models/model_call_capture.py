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
"""Model call capture: the middleware that records each model call a model server handles, and the store it writes to."""

import asyncio
import fcntl
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, NotRequired, Optional, TypedDict
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import orjson
from fastapi import Request, Response
from pydantic import BaseModel, model_validator

from nemo_gym.config.config_types import ROLLOUT_PATH_PREFIX, TOKEN_CAPTURE_PATH_SEGMENT
from nemo_gym.core.episode import VERIFIER_INVOCATION_HEADER, rollout_context
from nemo_gym.core.responses_api_types import SSE_KEEPALIVE
from nemo_gym.models._checkpoint import (
    attach_capture_context,
    ledger_removal_refusal,
)
from nemo_gym.models.responses_streaming import (
    is_responses_lifecycle_chunk,
)
from nemo_gym.models.token_capture import (
    CaptureContext,
    installed_lineage_store,
    installed_token_sink,
    reset_token_sink,
    set_token_sink,
)
from nemo_gym.models.token_capture.config import token_id_capture_config
from nemo_gym.models.token_capture.control_routes import install_rollout_control_routes
from nemo_gym.models.token_capture.lineage import FileLineageStore, InMemoryLineageStore
from nemo_gym.models.token_capture.protocols import (
    CaptureLedger,
    LineageResolver,
    ParentResolutionMode,
)
from nemo_gym.models.token_capture.records import UNCOMMITTED_CALL_REASON
from nemo_gym.models.token_capture.store import make_token_store
from nemo_gym.server_runtime.request_handling.rollout_context import scope_with_routed_path


logger = logging.getLogger(__name__)


_SSE_KEEPALIVE = SSE_KEEPALIVE


class ModelExecutionOutcome(TypedDict):
    error_category: NotRequired[str]
    upstream_attempted: bool
    response_source: Literal["upstream", "local"] | None
    upstream_status_code: int | None
    local_response_reason: str | None


def start_model_execution(request: Request, *, upstream_attempted: bool) -> ModelExecutionOutcome:
    """Keep adapter-owned execution facts separate from the served response and capture settings."""
    outcome = ModelExecutionOutcome(
        upstream_attempted=upstream_attempted,
        response_source=None,
        upstream_status_code=None,
        local_response_reason=None,
    )
    request.state.nemo_gym_model_execution = outcome
    return outcome


class ModelCallCaptureConfig(BaseModel):
    """Run-wide model-call capture settings from Gym's global config."""

    observability_enabled: bool = False
    model_call_capture_dir: Optional[Path] = None

    @model_validator(mode="after")
    def validate_capture_dir(self) -> "ModelCallCaptureConfig":
        if not self.observability_enabled:
            return self
        if self.model_call_capture_dir is None:
            raise ValueError("model_call_capture_dir is required when observability_enabled=true")
        if not self.model_call_capture_dir.is_absolute():
            raise ValueError("model_call_capture_dir must be an absolute path")
        return self


_PENDING_SUFFIX = ".capture.pending"


def _pending_token(model_call_id: str) -> str:
    """A filename-safe, dot-free token; the dot is the separator `has_pending` globs on."""
    token = "".join(char if (char.isascii() and (char.isalnum() or char in "_-")) else "_" for char in model_call_id)
    return token or "unknown"


def _validate_rollout_id(rollout_id: str) -> str:
    if not rollout_id or any(not (char.isascii() and (char.isalnum() or char in "._-")) for char in rollout_id):
        raise ValueError(f"Invalid rollout id: {rollout_id!r}")
    return rollout_id


class CaptureStore:
    """Append-only, rollout-keyed JSONL sink for model exchanges."""

    def __init__(self, root: str | Path, *, create: bool = True) -> None:
        self._root = Path(root)
        if create:
            self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, rollout_id: str) -> Path:
        return self._root / f"{_validate_rollout_id(rollout_id)}.capture.jsonl"

    def incomplete_path_for(self, rollout_id: str) -> Path:
        return self._root / f"{_validate_rollout_id(rollout_id)}.capture.incomplete"

    def mark_incomplete(self, rollout_id: str) -> None:
        self.incomplete_path_for(rollout_id).touch(exist_ok=True)

    def is_incomplete(self, rollout_id: str) -> bool:
        return self.incomplete_path_for(rollout_id).exists()

    def pending_path_for(self, rollout_id: str, model_call_id: str) -> Path:
        return self._root / f"{_validate_rollout_id(rollout_id)}.{_pending_token(model_call_id)}{_PENDING_SUFFIX}"

    def mark_pending(self, rollout_id: str, model_call_id: str) -> None:
        """Claim a write before it is queued, so a reader can tell 'not yet' from 'never'."""
        try:
            self.pending_path_for(rollout_id, model_call_id).touch(exist_ok=True)
        except OSError:
            logger.warning("Could not mark model-call capture pending for rollout %s.", rollout_id, exc_info=True)

    def clear_pending(self, rollout_id: str, model_call_id: str) -> None:
        try:
            self.pending_path_for(rollout_id, model_call_id).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not clear model-call capture pending for rollout %s.", rollout_id, exc_info=True)

    def has_pending(self, rollout_id: str) -> bool:
        """True while any claimed write for this rollout has not finished.

        The literal dot after the rollout id keeps `1-0` from matching `1-0-retry1`.
        """
        pattern = f"{_validate_rollout_id(rollout_id)}.*{_PENDING_SUFFIX}"
        return any(self._root.glob(pattern))

    def clear_all_pending(self, rollout_id: str) -> None:
        pattern = f"{_validate_rollout_id(rollout_id)}.*{_PENDING_SUFFIX}"
        for marker in self._root.glob(pattern):
            marker.unlink(missing_ok=True)

    def record(self, rollout_id: str, exchange: dict[str, Any]) -> None:
        """Append one exchange and fsync (durable across a killed box).

        ``flock`` serializes appends to the same rollout across worker processes and threads while
        allowing independent rollouts to write concurrently. This does blocking file IO + fsync,
        so callers run it off the event loop (the capture middleware uses ``asyncio.to_thread``).
        """
        line = orjson.dumps(exchange, default=str, option=orjson.OPT_APPEND_NEWLINE)
        path = self.path_for(rollout_id)
        with path.open("ab") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def read(self, rollout_id: str) -> list[dict[str, Any]]:
        path = self.path_for(rollout_id)
        if not path.exists():
            return []
        exchanges: list[dict[str, Any]] = []
        # Stream line-by-line; a capture can be large (token-ids / logprobs).
        with path.open("rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    exchanges.append(orjson.loads(stripped))
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return exchanges

    def read_available(self, rollout_id: str) -> tuple[list[tuple[int, dict[str, Any]]], int]:
        """Read valid exchanges without letting one damaged line hide the rest."""
        path = self.path_for(rollout_id)
        if not path.exists():
            return [], 0
        exchanges: list[tuple[int, dict[str, Any]]] = []
        invalid_count = 0
        capture_index = 0
        with path.open("rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        exchange = orjson.loads(stripped)
                    except orjson.JSONDecodeError:
                        invalid_count += 1
                    else:
                        if isinstance(exchange, dict):
                            exchanges.append((capture_index, exchange))
                        else:
                            invalid_count += 1
                    capture_index += 1
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return exchanges, invalid_count


_OBSERVED_PATHS = {
    "/v1/responses": "responses",
    "/v1/chat/completions": "chat",
    "/v1/messages": "messages",
}


# A client may declare its persisted session ID for exact correlation with an AgentInvocation.
# It is correlation evidence, not authentication; absence is fail-safe.
_CLIENT_SESSION_HEADER = b"x-session-id"


_TERMINAL_SSE_LINES: dict[str, dict[bytes, str]] = {
    "responses": {
        b"event: response.completed": "complete",
        b"event: response.incomplete": "incomplete",
        b"event: response.failed": "error",
        b"event: error": "error",
    },
    "chat": {b"data: [DONE]": "complete", b"event: error": "error"},
    "messages": {b"event: message_stop": "complete", b"event: error": "error"},
}


def _headers_content_type(headers: list) -> bytes:
    for key, value in headers:
        if key.lower() == b"content-type":
            return value
    return b""


def _unique_request_header(headers: list, name: bytes) -> Optional[str]:
    values = {value.decode("latin-1") for key, value in headers if key.lower() == name and value}
    return values.pop() if len(values) == 1 else None


def _preserve_capture_prefix_on_redirect(
    message: dict[str, Any],
    *,
    capture_prefix: str,
    request_headers: list[tuple[bytes, bytes]],
) -> dict[str, Any]:
    """Keep rollout correlation on root-relative and same-origin redirects."""
    status = int(message.get("status") or 0)
    if not 300 <= status < 400:
        return message

    request_host = next(
        (value.decode("latin-1") for key, value in request_headers if key.lower() == b"host"),
        "",
    )
    headers = list(message.get("headers") or [])
    changed = False
    for index, (key, value) in enumerate(headers):
        if key.lower() != b"location":
            continue
        location = value.decode("latin-1")
        parts = urlsplit(location)
        if parts.netloc and parts.netloc != request_host:
            continue
        if not parts.path.startswith("/") or parts.path.startswith(f"{capture_prefix}/"):
            continue
        headers[index] = (
            key,
            urlunsplit(parts._replace(path=f"{capture_prefix}{parts.path}")).encode("latin-1"),
        )
        changed = True

    if not changed:
        return message
    return {**message, "headers": headers}


def _consume_terminal_sse_event(buffer: bytearray, dialect: str) -> Optional[str]:
    blocks = re.split(rb"\r?\n\r?\n", bytes(buffer))
    buffer[:] = blocks.pop()
    terminal_lines = _TERMINAL_SSE_LINES[dialect]
    for block in blocks:
        lines = block.splitlines()
        for line in lines:
            field, separator, value = line.partition(b":")
            normalized = field + b": " + value.lstrip() if separator else line
            if normalized in terminal_lines:
                return terminal_lines[normalized]
        if dialect == "chat":
            for line in lines:
                if not line.startswith(b"data:"):
                    continue
                try:
                    payload = json.loads(line[5:].lstrip())
                except Exception:
                    continue
                if isinstance(payload, dict) and payload.get("error") is not None:
                    return "error"
    return None


# Consumer side of the URL-prefix protocol: strip /ng-rollout/<id> before routing, key capture by
# <id>. The constant + producer (apply_rollout_prefix) are in server_utils.
_ROLLOUT_PATH_RE = re.compile(
    rf"^/{re.escape(ROLLOUT_PATH_PREFIX)}/(?P<rollout_id>[^/]+)"
    rf"(?:/(?P<token_capture>{re.escape(TOKEN_CAPTURE_PATH_SEGMENT)}))?"
    rf"(?P<rest>/.*)$"
)


def make_capture_store(config: ModelCallCaptureConfig) -> Optional[CaptureStore]:
    """Build a CaptureStore when observability is enabled; otherwise None."""
    if not config.observability_enabled:
        return None
    root = config.model_call_capture_dir
    assert root is not None  # enforced by ModelCallCaptureConfig
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
    if isinstance(exc, asyncio.CancelledError):
        return "cancelled"
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "timeout"
    if "conn" in name:
        return "connection"
    return "exception"


def _exception_http_details(exc: BaseException) -> tuple[Optional[int], bytes]:
    def read_attr(value: Any, name: str) -> Any:
        try:
            return getattr(value, name, None)
        except Exception:
            return None

    response = read_attr(exc, "response")
    status = read_attr(exc, "status")
    if not isinstance(status, int):
        status = read_attr(exc, "status_code")
    if not isinstance(status, int) and response is not None:
        status = read_attr(response, "status_code")

    body = read_attr(exc, "response_content")
    if body is None and response is not None:
        body = read_attr(response, "content")
        if body is None:
            body = read_attr(response, "text")
    if isinstance(body, str):
        body = body.encode()
    return (status if isinstance(status, int) else None, bytes(body) if isinstance(body, (bytes, bytearray)) else b"")


# --- SSE reconstruction: rebuild a final response object from a streamed body ---
def _parse_sse_events(raw: bytes) -> list[dict[str, Any]]:
    """Parse an SSE byte stream into its JSON ``data:`` payloads (best-effort; non-JSON skipped)."""
    events: list[dict[str, Any]] = []
    for block in re.split(r"\r?\n\r?\n", raw.decode("utf-8", errors="replace")):
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
    refusal_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    usage: Optional[dict[str, Any]] = None
    model: Optional[str] = None
    response_id: Optional[str] = None
    role = "assistant"
    finish_reason: Optional[str] = None
    saw_choice = False
    for chunk in events:
        model = chunk.get("model") or model
        if isinstance(chunk.get("id"), str):
            response_id = chunk["id"]
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
            if delta.get("refusal"):
                refusal_parts.append(delta["refusal"])
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
    if refusal_parts:
        message["refusal"] = "".join(refusal_parts)
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    result: dict[str, Any] = {
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if response_id is not None:
        result["id"] = response_id
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
    model_server_name: Optional[str],
    request_bytes: bytes,
    *,
    verifier_invocation_id: Optional[str] = None,
    client_session_id: Optional[str] = None,
    client_assistant_message_id: Optional[str] = None,
    rollout_id: str,
    model_call_id: str,
    started_at: float,
    completed_at: float,
    response_body: Any,
    status_code: Optional[int],
    error_category: Optional[str],
    latency_ms: float,
    ttft_ms: Optional[float] = None,
    response_raw: Optional[str] = None,
    execution: Optional[ModelExecutionOutcome] = None,
) -> None:
    """Append one exchange (success or failure). Best-effort: never raises."""
    request_body = None
    request_raw = None
    if request_bytes:
        try:
            parsed_request = json.loads(request_bytes)
            if isinstance(parsed_request, dict):
                request_body = parsed_request
            else:
                request_raw = request_bytes.decode("utf-8", errors="replace")
        except Exception:
            request_raw = request_bytes.decode("utf-8", errors="replace")

    try:
        exchange = {
            "model_call_id": model_call_id,
            "dialect": dialect,
            "model_ref": {"type": "responses_api_models", "name": model_server_name} if model_server_name else None,
            "started_at": started_at,
            "completed_at": completed_at,
            "latency_ms": round(latency_ms, 2),
            "latency_ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
            "status_code": status_code,
            "error_category": error_category,
            "request": request_body,
            "response": response_body,
        }
        if verifier_invocation_id is not None:
            exchange["verifier_invocation_id"] = verifier_invocation_id
        if client_session_id is not None:
            exchange["client_session_id"] = client_session_id
        if execution is not None:
            exchange.update(execution)
            if execution["local_response_reason"] == "context_length_exceeded":
                exchange["error_category"] = "context_length_exceeded"
        if client_assistant_message_id is not None:
            exchange["client_assistant_message_id"] = client_assistant_message_id
        if request_raw is not None:
            exchange["request_raw"] = request_raw
        if response_raw is not None:
            exchange["response_raw"] = response_raw
        store.record(rollout_id, exchange)
    except Exception:
        logger.warning("Model-call capture failed for one %s call.", dialect, exc_info=True)
        try:
            store.mark_incomplete(rollout_id)
        except Exception:
            logger.warning("Could not mark rollout %s capture as incomplete.", rollout_id, exc_info=True)


async def _fail_uncommitted_external_call(context: CaptureContext | None) -> None:
    """Record a failure when an admitted worker call returns without commit coordinates.

    A call the engine refused is skipped: ``capture_engine_refusal`` already wrote its definite reason.
    """
    if (
        context is None
        or not context.external_staging
        or context.capture_admission is None
        or context.committed
        or context.refused
        or (context.parent_resolution_mode == "source_history" and not context.intent_registered)
        # A checkpoint gate owns a deferred row: it publishes it when the response starts,
        # and a call whose response never starts made no answer to record, so it is not poisoned.
        or context.deferred_commit is not None
        or context.engine_refusal_recorded
        or context.lineage_store is None
    ):
        return
    ledger = context.lineage_store
    if not isinstance(ledger, CaptureLedger):
        logger.error(
            "Cannot poison uncommitted call %s for rollout %s: the lineage resolver is not a CaptureLedger.",
            context.model_call_id,
            context.rollout_id,
        )
        return
    try:
        # A disconnected caller must not cancel the write before it reaches the ledger.
        await asyncio.shield(ledger.record_failure(context.rollout_id, context.model_call_id, UNCOMMITTED_CALL_REASON))
    except Exception:
        logger.exception(
            "Could not poison uncommitted call %s for rollout %s.",
            context.model_call_id,
            context.rollout_id,
        )


class _CaptureMiddleware:
    """Pure-ASGI per-rollout capture.

    Always strips an optional ``/ng-rollout/<id>`` path prefix before routing (used as the capture
    key) so the prefix is a stable routing feature independent of capture.
    When ``store`` is set it buffers the request body and a copy of the response while forwarding both
    downstream unchanged, so it composes with streaming (SSE) responses -- it never consumes or rewraps
    the stream. SSE chunks are forwarded immediately except for the terminal event, which is released
    after the capture is durable. Every chunk is also buffered for post-hoc reassembly, so a very long
    stream is held in memory until it completes. When ``store`` is None it avoids buffering evaluation
    records, while still persisting external capture failures before terminal events are forwarded.
    """

    def __init__(
        self,
        app: Any,
        *,
        store: CaptureStore | None,
        model_server_name: str | None,
        token_store: Any = None,
        configured_sink: Any = None,
        lineage_store: LineageResolver | None = None,
        delta_records: bool = False,
        external_staging: bool = False,
        parent_resolution_mode: ParentResolutionMode = "fingerprint",
        token_capture_enabled: bool = False,
        assistant_message_header: bytes | None = None,
        non_generating_requests: frozenset[tuple[str, str]] = frozenset(),
    ) -> None:
        self._app = app
        self._store = store
        self._model_server_name = model_server_name
        self._assistant_message_header: bytes | None = (
            assistant_message_header.lower() if assistant_message_header else None
        )
        # This store records training tokens for correlated training-capture calls.
        self._token_store = token_store
        # Built from token_id_capture.sink, once, in this process.
        self._configured_sink = configured_sink
        self._lineage_store: LineageResolver | None = lineage_store
        self._delta_records = delta_records
        # A framework inference worker owns record staging; the lineage store
        # doubles as the per-rollout capture ledger.
        self._external_staging = external_staging
        self._parent_resolution_mode = parent_resolution_mode
        self._capture_ledger: CaptureLedger | None = (
            lineage_store if external_staging and isinstance(lineage_store, CaptureLedger) else None
        )
        # Capture may have no destination in this process.
        # A framework may stage records from its inference worker.
        # This process still resolves the capture identity.
        self._token_capture_enabled = token_capture_enabled
        self._non_generating_requests = non_generating_requests

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        path = scope.get("path", "")
        rollout_from_path: Optional[str] = None
        token_capture_requested = False
        capture_prefix = ""
        prefix_match = _ROLLOUT_PATH_RE.match(path)
        if prefix_match:
            rollout_from_path = prefix_match.group("rollout_id")
            token_capture_requested = prefix_match.group("token_capture") is not None
            capture_prefix = path[: prefix_match.start("rest")]
            path = prefix_match.group("rest")
            scope = scope_with_routed_path(scope, path)

        method = str(scope.get("method") or "").upper()
        dialect = _OBSERVED_PATHS.get(path)
        known_non_generating = (method, path) in self._non_generating_requests

        # State is shared with the handler even through middleware scope copies.
        request_state = scope.setdefault("state", {})

        # Forward when no active store needs this correlated endpoint.
        # The prefix is already stripped.
        # An unprefixed call is forwarded rather than mixed with unrelated calls under a shared key.
        # Prefer the configured sink.
        # Then use the installed sink.
        # Finally use the file store.
        # Configured sinks are built in each server process.
        # Launcher-installed sinks do not reach spawned workers.
        # Installed sinks are resolved for each request.
        token_sink = self._configured_sink or installed_token_sink() or self._token_store
        capture_wanted = token_capture_requested and (token_sink is not None or self._token_capture_enabled)
        if (
            capture_wanted
            and self._parent_resolution_mode == "source_history"
            and dialect is not None
            and dialect not in ("responses", "chat")
        ):
            await Response(
                content='{"detail":"Source-history capture currently requires Responses or Chat"}',
                status_code=422,
                media_type="application/json",
            )(scope, receive, send)
            return
        if token_capture_requested and dialect is None and token_sink is not None:
            marked_incomplete = False
            response_started = False

            async def _mark_unobserved_incomplete(reason: str) -> None:
                nonlocal marked_incomplete
                if marked_incomplete:
                    return
                marked_incomplete = True
                try:
                    await token_sink.mark_incomplete(rollout_from_path, "")
                    logger.warning(
                        f"Marked rollout {rollout_from_path} incomplete because unclassified model request "
                        f"{method} {path} {reason}. Declare the exact request on the model server only if its "
                        "response cannot contain policy-generated content."
                    )
                except Exception:
                    logger.warning(
                        f"Could not mark rollout {rollout_from_path} incomplete for unobserved path {path}.",
                        exc_info=True,
                    )

            async def _send_unobserved(message: dict[str, Any]) -> None:
                nonlocal response_started
                if message.get("type") == "http.response.start":
                    response_started = True
                    status = int(message.get("status") or 0)
                    safe_without_capture = status in {404, 405}
                    if not known_non_generating and not safe_without_capture:
                        await _mark_unobserved_incomplete(f"returned HTTP {status}")
                    message = _preserve_capture_prefix_on_redirect(
                        message,
                        capture_prefix=capture_prefix,
                        request_headers=list(scope.get("headers") or []),
                    )
                await send(message)

            try:
                with rollout_context(rollout_from_path):
                    await self._app(scope, receive, _send_unobserved)
            except asyncio.CancelledError:
                if not known_non_generating and not response_started:
                    await _mark_unobserved_incomplete("was cancelled before starting a response")
                raise
            except Exception:
                if not known_non_generating and not response_started:
                    await _mark_unobserved_incomplete("failed before starting a response")
                raise
            if not known_non_generating and not response_started:
                await _mark_unobserved_incomplete("completed without starting a response")
            return
        if (self._store is None and not capture_wanted) or rollout_from_path is None or dialect is None:
            # Publish the id to this handler's current_rollout_id() even on the plain
            # forward path -- this is the common case for a model server with capture
            # disabled, and the id is already known here from the prefix above.
            with rollout_context(rollout_from_path):
                await self._app(scope, receive, send)
            return

        rollout_id = rollout_from_path
        model_call_id = uuid4().hex
        verifier_invocation_id = _unique_request_header(
            scope.get("headers") or [], VERIFIER_INVOCATION_HEADER.encode("ascii")
        )
        client_session_id = _unique_request_header(scope.get("headers") or [], _CLIENT_SESSION_HEADER)
        client_assistant_message_id = (
            _unique_request_header(scope.get("headers") or [], self._assistant_message_header)
            if self._assistant_message_header is not None
            else None
        )

        # Give the model server a token sink keyed to this call.
        # The sink records token ids from the complete response.
        # Middleware cannot recover token ids from SSE.
        # The context exists even without a local destination.
        # External staging uses the identity resolved here.
        sink_token = None
        capture_context = None
        if capture_wanted:
            capture_context = CaptureContext(
                rollout_id=rollout_id,
                model_call_id=model_call_id,
                token_sink=token_sink,
                lineage_store=self._capture_ledger if self._external_staging else self._lineage_store,
                delta_records=self._delta_records,
                external_staging=self._external_staging,
                parent_resolution_mode=self._parent_resolution_mode,
                admitted_at=time.time(),
            )
            sink_token = set_token_sink(capture_context)
            attach_capture_context(capture_context)

        # Training-only capture has no evaluation record.
        # Persist capture failure before forwarding a terminal event or finishing a JSON response.
        if self._store is None:
            streaming = False
            sse_event_buffer = bytearray()

            async def _send_training_only(message: dict[str, Any]) -> None:
                nonlocal streaming
                if message.get("type") == "http.response.start":
                    streaming = _headers_content_type(message.get("headers") or []).startswith(b"text/event-stream")
                elif message.get("type") == "http.response.body":
                    terminal = None
                    if streaming:
                        sse_event_buffer.extend(message.get("body", b"") or b"")
                        terminal = _consume_terminal_sse_event(sse_event_buffer, dialect)
                    if terminal is not None or not message.get("more_body", False):
                        await _fail_uncommitted_external_call(capture_context)
                await send(message)

            try:
                with rollout_context(rollout_id):
                    await self._app(scope, receive, _send_training_only)
            finally:
                await _fail_uncommitted_external_call(capture_context)
                if sink_token is not None:
                    reset_token_sink(sink_token)
            return

        request_body = bytearray()

        async def _receive() -> dict[str, Any]:
            message = await receive()
            if message.get("type") == "http.request":
                request_body.extend(message.get("body", b"") or b"")
            return message

        state: dict[str, Any] = {
            "status": None,
            "streaming": False,
            "body": bytearray(),
            "ttft_ms": None,
            "stream_terminal": None,
        }
        started_at = time.time()
        start = time.perf_counter()
        deferred_response_messages: list[dict[str, Any]] = []
        sse_event_buffer = bytearray()
        defer_response = False

        async def _send(message: dict[str, Any]) -> None:
            nonlocal defer_response
            message_type = message.get("type")
            if message_type == "http.response.start":
                state["status"] = message.get("status")
                content_type = _headers_content_type(message.get("headers") or [])
                state["streaming"] = content_type.startswith(b"text/event-stream")
            elif message_type == "http.response.body":
                chunk = message.get("body", b"") or b""
                if (
                    chunk
                    and chunk != _SSE_KEEPALIVE
                    # Lifecycle events open a `stream_keepalive_s` stream long before the first output item.
                    and not is_responses_lifecycle_chunk(chunk)
                    and state["ttft_ms"] is None
                ):
                    state["ttft_ms"] = (time.perf_counter() - start) * 1000.0
                state["body"].extend(chunk)  # buffered for both shapes; SSE is reassembled below
                if state["streaming"] and chunk and not defer_response:
                    sse_event_buffer.extend(chunk)
                    terminal = _consume_terminal_sse_event(sse_event_buffer, dialect)
                    defer_response = terminal is not None
                    if terminal is not None:
                        state["stream_terminal"] = terminal
                if defer_response or not message.get("more_body", False):
                    deferred_response_messages.append(dict(message))
                    return
            await send(message)  # forward unchanged -> streaming is preserved

        async def _flush_deferred_response() -> None:
            for message in deferred_response_messages:
                await send(message)

        # Claimed before the call is forwarded, for its whole lifetime. A claim taken only at
        # write time does not exist while the call is in flight, so a reader can look, see nothing
        # outstanding, and snapshot the rollout before the row lands.
        self._store.mark_pending(rollout_id, model_call_id)
        try:
            with rollout_context(rollout_id):
                await self._app(scope, _receive, _send)
        except (Exception, asyncio.CancelledError) as exc:
            completed_at = time.time()
            exception_status, exception_body = _exception_http_details(exc)
            upstream_status = state["status"] or exception_status
            upstream_body = bytes(state["body"]) or exception_body
            error_category = _classify_status(upstream_status) if isinstance(upstream_status, int) else None
            error_category = error_category or _classify_exception(exc)
            # Offload the blocking write+fsync so it never stalls the event loop.
            try:
                await asyncio.to_thread(
                    _record,
                    self._store,
                    dialect,
                    self._model_server_name,
                    bytes(request_body),
                    verifier_invocation_id=verifier_invocation_id,
                    client_session_id=client_session_id,
                    client_assistant_message_id=client_assistant_message_id,
                    rollout_id=rollout_id,
                    model_call_id=model_call_id,
                    started_at=started_at,
                    completed_at=completed_at,
                    response_body=None,
                    status_code=upstream_status,
                    error_category=error_category,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    ttft_ms=state["ttft_ms"],
                    response_raw=upstream_body.decode("utf-8", errors="replace") if upstream_body else None,
                    execution=request_state.get("nemo_gym_model_execution"),
                )
            except Exception:
                logger.warning("Model-call capture finalization failed.", exc_info=True)
            finally:
                self._store.clear_pending(rollout_id, model_call_id)
                await _fail_uncommitted_external_call(capture_context)
                await _flush_deferred_response()
            raise
        finally:
            # The sink is only needed while the model server produces the response.
            await _fail_uncommitted_external_call(capture_context)
            if sink_token is not None:
                reset_token_sink(sink_token)

        completed_at = time.time()
        latency_ms = (time.perf_counter() - start) * 1000.0
        status = state["status"]
        body_bytes = bytes(state["body"])
        streaming = state["streaming"]
        stream_terminal = state["stream_terminal"]
        ttft_ms = state["ttft_ms"]
        request_bytes = bytes(request_body)
        store, model_server_name = self._store, self._model_server_name

        def _parse_and_record() -> None:
            # Off the event loop: body parse + SSE reassembly is best-effort and fully guarded, so a
            # malformed body can never surface as an ASGI error after the response was already sent.
            response_body = None
            if body_bytes:
                try:
                    response_body = (
                        _reconstruct_streamed_response(body_bytes, dialect) if streaming else json.loads(body_bytes)
                    )
                    if not isinstance(response_body, dict):
                        response_body = None
                    elif streaming and "nemo_gym_internal_usage" in request_state:
                        # Keep unknown detail counts as None, as non-streaming capture does.
                        response_body["usage"] = request_state["nemo_gym_internal_usage"]
                except Exception:
                    response_body = None
            error_category = _classify_status(status) if status is not None else None
            if error_category is None:
                error_category = {
                    "error": "upstream_error",
                    "incomplete": "incomplete",
                }.get(stream_terminal)
            if error_category is None and streaming and stream_terminal is None:
                error_category = "stream_truncated"
            # A 2xx whose body we couldn't parse/reassemble isn't a clean success -- flag it so it
            # doesn't silently count as a success with null tokens in reliability/cost sums.
            if error_category is None and body_bytes and response_body is None:
                error_category = "capture_parse_error"
            response_raw = (
                body_bytes.decode("utf-8", errors="replace")
                if body_bytes and (streaming or response_body is None)
                else None
            )
            try:
                _record(
                    store,
                    dialect,
                    model_server_name,
                    request_bytes,
                    verifier_invocation_id=verifier_invocation_id,
                    client_session_id=client_session_id,
                    client_assistant_message_id=client_assistant_message_id,
                    rollout_id=rollout_id,
                    model_call_id=model_call_id,
                    started_at=started_at,
                    completed_at=completed_at,
                    response_body=response_body,
                    status_code=status,
                    error_category=error_category,
                    latency_ms=latency_ms,
                    ttft_ms=ttft_ms,
                    response_raw=response_raw,
                    execution=request_state.get("nemo_gym_model_execution"),
                )
            finally:
                store.clear_pending(rollout_id, model_call_id)

        try:
            # Shielded: cancellation here strands the queued write and the row is
            # lost silently -- no log line, no incomplete marker. The error path
            # above already treats CancelledError as recoverable; match it.
            await asyncio.shield(asyncio.to_thread(_parse_and_record))
        except asyncio.CancelledError:
            logger.warning("Model-call capture finalization was cancelled; the write continues detached.")
            raise
        except Exception:
            logger.warning("Model-call capture finalization failed.", exc_info=True)
        finally:
            await _flush_deferred_response()


def install_model_call_capture(
    app: Any,
    config: ModelCallCaptureConfig,
    *,
    model_server_name: str | None = None,
    assistant_message_header: bytes | None = None,
    global_config_dict: Any = None,
    num_workers: int | None = None,
    non_generating_requests: frozenset[tuple[str, str]] = frozenset(),
) -> CaptureLedger | None:
    """Install model-call capture middleware and return the capture ledger when staging is external.

    Always strip ``/ng-rollout/<id>/...`` before routing.
    Evaluation capture records requests and responses for that path.
    Non-terminal SSE chunks continue immediately.
    The terminal event follows the durable evaluation write.
    Training capture uses ``/ng-rollout/<id>/training-token-capture/...``.
    That path provides a request-scoped token sink.
    The model server records token ids from its complete response.
    Consumers access records through ``TokenSource.freeze``.
    """
    capture_settings = token_id_capture_config(global_config_dict) if global_config_dict is not None else None
    token_store = make_token_store(global_config_dict) if global_config_dict is not None else None
    # Build this sink at app startup.
    # Each uvicorn worker constructs its own sink.
    # Spawned workers do not inherit a launcher-installed sink.
    configured_sink = capture_settings.build_sink() if capture_settings is not None else None
    configured_lineage = capture_settings.build_lineage_store() if capture_settings is not None else None
    lineage_store = configured_lineage or installed_lineage_store()
    default_lineage = None
    if lineage_store is None and token_store is not None:
        default_lineage = FileLineageStore(token_store.root)
        lineage_store = default_lineage
    if capture_settings is not None and capture_settings.enabled and (num_workers or 1) > 1:
        if lineage_store is None or not lineage_store.is_process_shared():
            raise ValueError(
                "token_id_capture with num_workers > 1 requires a process-shared lineage resolver "
                "over the same backend used by the token sink"
            )
    external_staging = capture_settings is not None and capture_settings.token_id_capture.external_staging
    if external_staging:
        if lineage_store is None:
            raise ValueError(
                "token_id_capture.external_staging requires a lineage resolver that implements CaptureLedger"
            )
        if isinstance(lineage_store, InMemoryLineageStore):
            # The in-memory index evicts rollouts under memory bounds, which is
            # acceptable for a resolution cache but not for a completeness
            # ledger. External staging requires a non-evicting store.
            raise ValueError(
                "token_id_capture.external_staging requires a non-evicting lineage store "
                "(e.g. FileLineageStore); InMemoryLineageStore cannot serve as the capture ledger"
            )
        if not isinstance(lineage_store, CaptureLedger):
            raise ValueError(
                "token_id_capture.external_staging requires the lineage store to implement "
                "the CaptureLedger protocol (record, record_failure, manifest, has_rows, retire, delete)"
            )
        capture_ledger: CaptureLedger = lineage_store
        install_rollout_control_routes(
            app,
            capture_ledger,
            auth_token=capture_settings.token_id_capture.resolve_control_auth_token(),
            refuse_removal=lambda: ledger_removal_refusal(app),
        )

    owned_endpoints = [
        endpoint
        for endpoint in (configured_sink, token_store, configured_lineage, default_lineage)
        if endpoint is not None
    ]

    async def _close_capture_endpoints() -> None:
        for endpoint in owned_endpoints:
            await endpoint.close()

    if owned_endpoints:
        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def _capture_lifespan(application):
            try:
                async with original_lifespan(application) as state:
                    yield state
            finally:
                await _close_capture_endpoints()

        app.router.lifespan_context = _capture_lifespan
    app.add_middleware(
        _CaptureMiddleware,
        store=make_capture_store(config),
        assistant_message_header=assistant_message_header,
        model_server_name=model_server_name,
        token_store=token_store,
        configured_sink=configured_sink,
        lineage_store=lineage_store,
        delta_records=(capture_settings.token_id_capture.delta_records if capture_settings is not None else False),
        external_staging=external_staging,
        parent_resolution_mode=(
            capture_settings.token_id_capture.parent_resolution if capture_settings is not None else "fingerprint"
        ),
        token_capture_enabled=capture_settings.enabled if capture_settings is not None else False,
        non_generating_requests=non_generating_requests,
    )
    return capture_ledger if external_staging else None
