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
"""Model server base classes and per-rollout model-call capture.

Every Gym model server derives from ``SimpleResponsesAPIModel``, which wires the three model
dialects (/v1/responses, /v1/chat/completions, /v1/messages) and installs the model-call capture
middleware.

Capture is opt-in, off by default. A pure-ASGI middleware records correlated /v1/responses,
/v1/chat/completions, and /v1/messages exchanges -- including failed calls -- into a
per-rollout CaptureStore, forwarding bytes downstream unchanged so it composes with
streaming (SSE) responses. Best-effort; never alters the response. Correlation is
carried by a /ng-rollout/<rollout_id>/v1/... base_url prefix, which is stripped before
routing.
"""

import asyncio
import fcntl
import inspect
import json
import logging
import os
import re
import threading
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import orjson
from fastapi import Body, FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, model_validator

from nemo_gym.anthropic_converter import AnthropicConverter
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import (
    ROLLOUT_PATH_PREFIX,
    BaseRunServerInstanceConfig,
    BaseServer,
    SimpleServer,
    maybe_rollout_id_from_run_body,
)


logger = logging.getLogger(__name__)


# Stateless; shared by every model server's default /v1/messages handler.
_ANTHROPIC_CONVERTER = AnthropicConverter()


class BaseResponsesAPIModelConfig(BaseRunServerInstanceConfig):
    pass


class BaseResponsesAPIModel(BaseServer):
    config: BaseResponsesAPIModelConfig


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


class SimpleResponsesAPIModel(BaseResponsesAPIModel, SimpleServer):
    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)
        capture_config = ModelCallCaptureConfig.model_validate(self.server_client.global_config_dict)
        self.install_model_call_capture(app, capture_config)

        app.post("/v1/chat/completions")(self.chat_completions)

        app.post("/v1/responses")(self.responses)

        # Every Gym model server speaks the Anthropic Messages API by default, mapping
        # Messages <-> Responses around its own responses() implementation. This lets blackbox
        # harnesses that require an Anthropic endpoint (e.g. the Claude Code CLI) target any
        # model server directly.
        app.post("/v1/messages")(self.messages)

        return app

    def install_model_call_capture(self, app: Any, capture_config: ModelCallCaptureConfig) -> None:
        """Install model-call capture middleware.

        Always installed so the ``/ng-rollout/<id>`` correlation prefix is stripped before routing
        regardless of whether capture is enabled (otherwise a default ``gym eval`` would 404 on every
        prefixed model call). When capture is enabled the middleware additionally records each observed
        call's request + response into a rollout-keyed CaptureStore while forwarding bytes downstream
        unchanged (non-terminal SSE chunks are forwarded as they arrive; the terminal event follows the
        durable capture write).
        """
        if not capture_config.observability_enabled:
            return

        app.add_middleware(
            _CaptureMiddleware,
            store=CaptureStore(capture_config.model_call_capture_dir),
            model_server_name=self.config.name,
        )

    @abstractmethod
    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        pass

    @abstractmethod
    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        pass

    async def messages(self, request: Request, body: dict = Body()):
        """Default Anthropic Messages <-> Responses mapping shared by every Gym model server.

        Translates the inbound Anthropic Messages request to the Responses API, delegates to this
        server's own ``responses()`` (so it reuses whatever backend the server has), and maps the
        result back to an Anthropic Messages response. When the client requested ``stream: true``
        (the Claude Code CLI always does), the complete response is re-emitted as a synthesized
        Anthropic SSE event stream. Servers may override this for native Messages handling.
        """
        params = _ANTHROPIC_CONVERTER.anthropic_request_to_responses(body)
        response = await self._invoke_responses(request, params)
        model_name = body.get("model") or response.model
        anthropic_response = _ANTHROPIC_CONVERTER.responses_to_anthropic_response(response, model=model_name)
        if body.get("stream"):
            return StreamingResponse(
                _ANTHROPIC_CONVERTER.anthropic_response_to_sse(anthropic_response),
                media_type="text/event-stream",
            )
        return anthropic_response

    async def _invoke_responses(
        self, request: Request, params: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymResponse:
        # responses() signatures vary across servers: some take a leading `request`, some only
        # `body`. Dispatch on whichever this server declares so the default messages() works for
        # all of them.
        if "request" in inspect.signature(self.responses).parameters:
            return await self.responses(request=request, body=params)
        return await self.responses(body=params)


# --- Capture configuration + rollout-keyed storage ---


def _validate_rollout_id(rollout_id: str) -> str:
    if not rollout_id or any(not (char.isascii() and (char.isalnum() or char in "._-")) for char in rollout_id):
        raise ValueError(f"Invalid rollout id: {rollout_id!r}")
    return rollout_id


class CaptureStore:
    """Append-only, rollout-keyed JSONL sink for model exchanges."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, rollout_id: str) -> Path:
        return self._root / f"{_validate_rollout_id(rollout_id)}.capture.jsonl"

    def record(self, rollout_id: str, exchange: dict[str, Any]) -> None:
        """Append one exchange and fsync (durable across a killed box).

        ``flock`` serializes appends across worker processes (a model server may run with
        ``num_workers > 1``, where the in-process lock can't coordinate); the in-process lock
        serializes threads. This does blocking file IO + fsync, so callers run it off the event
        loop (the capture middleware offloads it via ``asyncio.to_thread``).
        """
        line = orjson.dumps(exchange, default=str, option=orjson.OPT_APPEND_NEWLINE)
        path = self.path_for(rollout_id)
        with self._lock:
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
        with self._lock:
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


# --- Observability records derived from captured exchanges ---


class ModelCallRecord(BaseModel):
    """Observability record derived from one captured model-server exchange."""

    # HTTP information
    status_code: int
    route: str

    # Gym information
    model_ref: ModelServerRef

    # Model-call record
    request: NeMoGymResponseCreateParamsNonStreaming
    response: Optional[NeMoGymResponse]  # Only present if the call succeeded

    # Used for cases where we never hit a NeMoGymResponsesCreateParams or NeMoGymResponse in a model call e.g. calling an Anthropic model with /v1/messages
    # For those scenarios we always store the raw_request and raw_response and provided a normalized version by converting to Responses
    # For normal Responses routes, this is empty.
    raw_request: Optional[Dict[str, Any]]
    raw_response: Optional[Dict[str, Any]]


def read_model_call_records(store: CaptureStore, rollout_id: str) -> list[ModelCallRecord]:
    """Read captured exchanges in durable append order."""
    return store.read(rollout_id)


def aggregate_model_call_records(calls: list[ModelCallRecord]) -> dict[str, Any]:
    """Aggregate token and latency values from model-call records."""

    def _sum(attr: str) -> Optional[float]:
        values = [getattr(call, attr) for call in calls if getattr(call, attr) is not None]
        return sum(values) if values else None

    return {
        "tokens_in": _sum("tokens_in"),
        "tokens_out": _sum("tokens_out"),
        "tokens_reasoning": _sum("tokens_reasoning"),
        "tokens_total": _sum("tokens_total"),
        "latency_total_ms": _sum("latency_total_ms"),
        "num_calls": len(calls),
    }


def aggregate_model_call_metrics(store: CaptureStore, rollout_id: str) -> dict[str, Any]:
    """Aggregate model-call metrics for one rollout id."""
    return aggregate_model_call_records(read_model_call_records(store, rollout_id))


# --- Capture middleware ---


_OBSERVED_PATHS = {
    "/v1/responses": "responses",
    "/v1/chat/completions": "chat",
    "/v1/messages": "messages",
}


def _headers_content_type(headers: list) -> bytes:
    for key, value in headers:
        if key.lower() == b"content-type":
            return value
    return b""


# Consumer side of the URL-prefix protocol: strip /ng-rollout/<id> before routing, key capture by
# <id>. The constant + producer (apply_rollout_prefix) are in server_utils.
_ROLLOUT_PATH_RE = re.compile(rf"^/{re.escape(ROLLOUT_PATH_PREFIX)}/(?P<rollout_id>[^/]+)(?P<rest>/.*)$")


def _record(
    store: CaptureStore,
    dialect: str,
    model_server_name: Optional[str],
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
    store.record(
        rollout_id,
        {
            "dialect": dialect,
            "model_server": model_server_name,
            "latency_ms": round(latency_ms, 2),
            "latency_ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
            "status_code": status_code,
            "error_category": error_category,
            "request": json.loads(request_bytes) if request_bytes else None,
            "response": response_body,
        },
    )


class _CaptureMiddleware:
    """Pure-ASGI per-rollout capture.

    Always strips an optional ``/ng-rollout/<id>`` path prefix before routing (used as the capture
    key) so the prefix is a stable routing feature independent of capture.
    When ``store`` is set it buffers the request body and a copy of the response while forwarding both
    downstream unchanged, so it composes with streaming (SSE) responses -- it never consumes or rewraps
    the stream. SSE chunks are forwarded immediately except for the terminal event, which is released
    after the capture is durable. Every chunk is also buffered for post-hoc reassembly, so a very long
    stream is held in memory until it completes. When ``store`` is None (capture disabled) it strips the
    prefix and forwards only.
    """

    def __init__(self, app: Any, *, store: Optional[CaptureStore], model_server_name: Optional[str]) -> None:
        self._app = app
        self._store = store
        self._model_server_name = model_server_name

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

        # Only explicitly correlated model calls are captured. An unprefixed call is forwarded
        # unchanged rather than being mixed with unrelated calls under a shared fallback key.
        if rollout_from_path is None:
            await self._app(scope, receive, send)
            return

        dialect = _OBSERVED_PATHS.get(path)
        if dialect is None:
            await self._app(scope, receive, send)  # not observed (or a stripped non-/v1 path)
            return

        rollout_id = rollout_from_path
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
            "stream_error_category": None,
        }
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
                if chunk and state["ttft_ms"] is None:
                    state["ttft_ms"] = (time.perf_counter() - start) * 1000.0
                state["body"].extend(chunk)  # buffered for both shapes; SSE is reassembled below
                if state["streaming"] and chunk and not defer_response:
                    sse_event_buffer.extend(chunk)
                    terminal = None
                    defer_response = terminal is not None
                    state["stream_error_category"] = {
                        "error": "upstream_error",
                        "incomplete": "incomplete",
                    }.get(terminal)
                if defer_response or not message.get("more_body", False):
                    deferred_response_messages.append(dict(message))
                    return
            await send(message)  # forward unchanged -> streaming is preserved

        async def _flush_deferred_response() -> None:
            for message in deferred_response_messages:
                await send(message)

        try:
            await self._app(scope, _receive, _send)
        except Exception:
            # Offload the blocking write+fsync so it never stalls the event loop.
            try:
                await asyncio.to_thread(
                    _record,
                    self._store,
                    dialect,
                    self._model_server_name,
                    bytes(request_body),
                    rollout_id=rollout_id,
                    response_body=None,
                    status_code=None,
                    error_category=None,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    ttft_ms=state["ttft_ms"],
                )
            except Exception:
                logger.warning("Model-call capture finalization failed.", exc_info=True)
            finally:
                await _flush_deferred_response()
            raise

        latency_ms = (time.perf_counter() - start) * 1000.0
        status = state["status"]
        body_bytes = bytes(state["body"])
        stream_error_category = state["stream_error_category"]
        ttft_ms = state["ttft_ms"]
        request_bytes = bytes(request_body)
        store, model_server_name = self._store, self._model_server_name

        def _parse_and_record() -> None:
            # Off the event loop: body parse + SSE reassembly is best-effort and fully guarded, so a
            # malformed body can never surface as an ASGI error after the response was already sent.
            response_body = None
            if body_bytes:
                response_body = orjson.loads(body_bytes)
            error_category = None
            if error_category is None and stream_error_category:
                error_category = stream_error_category
            # A 2xx whose body we couldn't parse/reassemble isn't a clean success -- flag it so it
            # doesn't silently count as a success with null tokens in reliability/cost sums.
            if error_category is None and body_bytes and response_body is None:
                error_category = "capture_parse_error"
            _record(
                store,
                dialect,
                model_server_name,
                request_bytes,
                rollout_id=rollout_id,
                response_body=response_body,
                status_code=status,
                error_category=error_category,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
            )

        try:
            await asyncio.to_thread(_parse_and_record)
        except Exception:
            logger.warning("Model-call capture finalization failed.", exc_info=True)
        finally:
            await _flush_deferred_response()


# --- Run-level capture helpers (rollout-collection side) ---


def model_call_capture_dirs_from_config(global_config_dict: Any) -> list[Path]:
    """Return the single run-wide capture directory when capture is enabled."""
    config = ModelCallCaptureConfig.model_validate(global_config_dict)
    if not config.observability_enabled:
        return []
    assert config.model_call_capture_dir is not None  # enforced by ModelCallCaptureConfig
    return [config.model_call_capture_dir]


def _store_for_rollout(rollout_id: str, capture_dirs: list[Path]) -> Optional[CaptureStore]:
    for directory in capture_dirs:
        store = CaptureStore(directory)
        if store.path_for(rollout_id).exists():
            return store
    return None


def clear_model_call_captures_for_rollouts(records: list[Any], capture_dirs: list[Path]) -> None:
    """Remove stale per-rollout capture files for these records before a fresh (non-resume) run.

    Capture files are keyed by a deterministic rollout id (task-rollout-attempt), so without this a
    re-run would append onto the previous run's capture for the same id. This run-scopes the capture
    so each run's model-call evidence stays isolated.
    """
    if not capture_dirs:
        return
    for directory in capture_dirs:
        store = CaptureStore(directory)
        for record in records:
            rollout_id = maybe_rollout_id_from_run_body(record)
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
    rollout_id = maybe_rollout_id_from_run_body(record)
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
