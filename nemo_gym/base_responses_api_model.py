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
from typing import Any, Mapping, Optional
from uuid import uuid4

import orjson
from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError, model_validator

from nemo_gym.anthropic_converter import AnthropicConverter
from nemo_gym.config_types import ROLLOUT_PATH_PREFIX, ModelServerRef
from nemo_gym.global_config import (
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.responses_streaming import (
    sanitize_streaming_responses_body,
    synthesize_responses_failure_sse,
    synthesize_responses_sse,
    validate_streaming_responses_params,
)
from nemo_gym.server_utils import (
    BaseRunServerInstanceConfig,
    BaseServer,
    SimpleServer,
)


logger = logging.getLogger(__name__)


# Stateless; shared by every model server's default /v1/messages handler.
_ANTHROPIC_CONVERTER = AnthropicConverter()


class BaseResponsesAPIModelConfig(BaseRunServerInstanceConfig):
    pass


class BaseResponsesAPIModel(BaseServer):
    config: BaseResponsesAPIModelConfig


class SimpleResponsesAPIModel(BaseResponsesAPIModel, SimpleServer):
    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)
        capture_config = ModelCallCaptureConfig.model_validate(self.server_client.global_config_dict)
        install_model_call_capture(app, capture_config, model_server_name=self.config.name)

        app.post("/v1/chat/completions")(self.chat_completions)

        app.post("/v1/responses")(self.responses_dispatch)

        # Every Gym model server speaks the Anthropic Messages API by default, mapping
        # Messages <-> Responses around its own responses() implementation. This lets blackbox
        # harnesses that require an Anthropic endpoint (e.g. the Claude Code CLI) target any
        # model server directly.
        app.post("/v1/messages")(self.messages)

        return app

    @abstractmethod
    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        pass

    @abstractmethod
    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        pass

    async def responses_dispatch(self, request: Request, body: dict = Body()):
        """Default ``/v1/responses`` entrypoint shared by every Gym model server.

        A plain JSON request validates strictly against
        ``NeMoGymResponseCreateParamsNonStreaming`` and delegates to this server's own
        ``responses()``, preserving the historical non-streaming behavior. When the client
        requests ``stream: true`` (blackbox Responses-over-SSE harnesses like the Codex CLI
        always do), the request is first sanitized from the streaming wire dialect (extra
        bookkeeping fields, ``namespace`` tool specs — see ``nemo_gym.responses_streaming``),
        delegated to the same ``responses()``, and the complete response is re-emitted as a
        synthesized Responses SSE event stream. A ``responses()`` failure on this path is turned
        into a terminal ``response.failed`` event rather than an HTTP 500 (bad-request validation
        still fails eagerly, before the stream is committed).
        """
        if not body.get("stream"):
            params = _validate_responses_params(body)
            return await self._invoke_responses(request, params)

        cleaned, ns_map = sanitize_streaming_responses_body(body)
        try:
            params = validate_streaming_responses_params(cleaned)
        except ValidationError as exc:
            raise RequestValidationError([{**error, "loc": ("body", *error["loc"])} for error in exc.errors()])

        try:
            response = await self._invoke_responses(request, params)
            response_json = response.model_dump(mode="json") if isinstance(response, BaseModel) else dict(response)
        except Exception as exc:
            # The streaming contract is already the response's shape, so a backend failure must be a
            # terminal response.failed event, not an HTTP 500 the client would see as a broken stream.
            logger.exception("responses() failed while serving a streaming /v1/responses request")
            return StreamingResponse(
                synthesize_responses_failure_sse(str(exc)),
                media_type="text/event-stream",
            )
        return StreamingResponse(
            synthesize_responses_sse(response_json, ns_map),
            media_type="text/event-stream",
        )

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


def _validate_responses_params(body: dict) -> NeMoGymResponseCreateParamsNonStreaming:
    """Validate a /v1/responses body dict, surfacing failures as FastAPI's standard 422."""
    try:
        return NeMoGymResponseCreateParamsNonStreaming.model_validate(body)
    except ValidationError as exc:
        raise RequestValidationError([{**error, "loc": ("body", *error["loc"])} for error in exc.errors()])


# --- Capture configuration + rollout-keyed storage ---


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


def maybe_rollout_id_from_run_body(body: BaseModel | Mapping[str, Any] | None) -> Optional[str]:
    """Per-rollout model-call capture id from a run-request's task/rollout indices.

    Reads the canonical row keys (``_ng_task_index`` / ``_ng_rollout_index``) that
    rollout_collection ships to an agent's ``/run``. When a resume re-dispatch attempt is present
    (``_ng_attempt_index`` > 0), an ``-a<n>`` suffix is appended so a retry's captured model calls
    stay separable from the prior attempt; the first attempt (0) keeps the bare ``<task>-<rollout>``
    key for backward compatibility.
    """
    if isinstance(body, BaseModel):
        data = body.model_dump()
    elif isinstance(body, Mapping):
        data = body
    else:
        return None
    task = data.get(TASK_INDEX_KEY_NAME)
    rollout = data.get(ROLLOUT_INDEX_KEY_NAME)
    if task is None or rollout is None:
        return None
    rollout_id = f"{task}-{rollout}"
    attempt = data.get(ATTEMPT_INDEX_KEY_NAME)
    if attempt is not None:
        attempt_index = int(attempt)
        if attempt_index > 0:
            rollout_id = f"{rollout_id}-a{attempt_index}"
    return rollout_id


# --- Observability records derived from captured exchanges ---


def extract_token_stats(usage: Any) -> dict[str, Optional[int]]:
    """Normalize token totals across Responses, Chat Completions, and Anthropic Messages usage.

    For native Anthropic ``/v1/messages`` with prompt caching, ``input_tokens`` is only the uncached
    remainder, so cache-read + cache-creation tokens are folded into ``tokens_in`` to reflect the true
    prompt size (and cache-creation is surfaced separately as ``cache_creation_tokens``). OpenAI /
    Responses usage already includes cached tokens in ``input_tokens`` / ``prompt_tokens`` (where
    ``cached_tokens`` is a subset), so it is left untouched -- no double counting.

    ``tokens_in`` is a prompt-*size* metric, not a cost proxy: providers price cache-read (~0.1x) and
    cache-creation (~1.25x) differently from base input, so cost-accurate consumers should weight
    ``cached_tokens`` and ``cache_creation_tokens`` separately rather than summing ``tokens_in``.
    """
    if not usage:
        return {
            "tokens_in": None,
            "tokens_out": None,
            "tokens_reasoning": None,
            "tokens_total": None,
            "cache_creation_tokens": None,
        }
    tokens_in = usage.get("input_tokens")
    if tokens_in is None:
        tokens_in = usage.get("prompt_tokens")
    tokens_out = usage.get("output_tokens")
    if tokens_out is None:
        tokens_out = usage.get("completion_tokens")
    # Anthropic-native shape: top-level cache_* keys mean input_tokens excludes cached tokens.
    cache_read = usage.get("cache_read_input_tokens")
    cache_creation = usage.get("cache_creation_input_tokens")
    if cache_read is not None or cache_creation is not None:
        # A fully-cached response can omit input_tokens; use a 0 base so the folded prompt size is
        # preserved rather than dropped to null. (Top-level cache_* keys are Anthropic-only, so the
        # OpenAI/Responses path -- nested prompt_tokens_details.cached_tokens -- never enters here.)
        tokens_in = (tokens_in or 0) + (cache_read or 0) + (cache_creation or 0)
    tokens_total = usage.get("total_tokens")
    if tokens_total is None and tokens_in is not None and tokens_out is not None:
        tokens_total = tokens_in + tokens_out
    details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    return {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_reasoning": details.get("reasoning_tokens"),
        "tokens_total": tokens_total,
        "cache_creation_tokens": cache_creation,
    }


def _cache_signal(usage: Any) -> tuple[Optional[bool], Optional[int]]:
    """Cache hit/miss + cached-token count, from usage cache fields (OpenAI / Anthropic)."""
    if not usage:
        return None, None
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is None:
        cached = usage.get("cache_read_input_tokens")  # Anthropic
    if cached is None:
        return None, None
    return cached > 0, cached


def _as_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except Exception:
            return {"_raw": arguments}
    return {}


def _tool_calls_and_reasoning(response: dict[str, Any]) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Structured tool calls (name, arguments, call_id) and reasoning text, across all three shapes."""
    tool_calls: list[dict[str, Any]] = []
    reasoning: list[str] = []

    output = response.get("output")
    if isinstance(output, list):  # Responses
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call":
                tool_calls.append(
                    {
                        "call_id": item.get("call_id") or item.get("id"),
                        "name": item.get("name"),
                        "arguments": _as_arguments(item.get("arguments")),
                    }
                )
            elif item.get("type") == "reasoning":
                for summary in item.get("summary") or []:
                    text = summary.get("text") if isinstance(summary, dict) else None
                    if text:
                        reasoning.append(text)
        return tool_calls, ("\n".join(reasoning) or None)

    choices = response.get("choices")
    if isinstance(choices, list):  # Chat Completions
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if not message:
                continue
            for tc in message.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                tool_calls.append(
                    {"call_id": tc.get("id"), "name": fn.get("name"), "arguments": _as_arguments(fn.get("arguments"))}
                )
            # vLLM and newer OpenAI-compatible servers emit `reasoning`; `reasoning_content` is the
            # older field. Accept either (reasoning_content wins when both are present).
            reasoning_text = message.get("reasoning_content") or message.get("reasoning")
            if reasoning_text:
                reasoning.append(reasoning_text)
        return tool_calls, ("\n".join(reasoning) or None)

    content = response.get("content")
    if isinstance(content, list):  # Anthropic Messages
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tool_calls.append(
                    {"call_id": block.get("id"), "name": block.get("name"), "arguments": block.get("input") or {}}
                )
            elif block.get("type") in ("thinking", "redacted_thinking") and block.get("thinking"):
                reasoning.append(block["thinking"])
        return tool_calls, ("\n".join(reasoning) or None)

    return tool_calls, None


class ModelCallRecord(BaseModel):
    """Observability record derived from one captured model-server exchange."""

    # Unique server-generated identity for each persisted call.
    model_call_id: Optional[str] = None

    # Durable append order, not a causal or semantic order for concurrent calls.
    call_index: int
    model_ref: Optional[ModelServerRef] = None
    dialect: Optional[str] = None
    status_code: Optional[int] = None

    # Wall-clock bounds around the downstream ASGI invocation, as UTC Unix timestamps. These are
    # for external trace correlation; durations use the monotonic latency fields below.
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Token accounting. tokens_reasoning is OpenAI/Responses-only
    # (sourced from *_tokens_details.reasoning_tokens); Anthropic does not expose it, so it is null
    # there -- consumers must treat null as "unknown", not 0.
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    tokens_reasoning: Optional[int] = None
    tokens_total: Optional[int] = None

    # Model-call record.
    request: Optional[dict[str, Any]] = None
    response: Optional[dict[str, Any]] = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)

    # Structured reasoning (not flattened into the response text).
    reasoning_content: Optional[str] = None

    # Cache visibility. cached_tokens is the cache-read count; cache_creation_tokens is the
    # Anthropic cache-write count (also folded into tokens_in for the true prompt size).
    cache_hit: Optional[bool] = None
    cached_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None

    # Error classification.
    error_category: Optional[str] = None

    # Latency.
    latency_total_ms: Optional[float] = None
    latency_ttft_ms: Optional[float] = None


def build_model_call_record(exchange: dict[str, Any], *, call_index: int) -> ModelCallRecord:
    """Map one captured exchange and its transport metadata into an observability record."""
    response = exchange.get("response") or {}
    tokens = extract_token_stats(response.get("usage"))
    cache_hit, cached_tokens = _cache_signal(response.get("usage"))
    tool_calls, reasoning_content = _tool_calls_and_reasoning(response)
    return ModelCallRecord(
        model_call_id=exchange.get("model_call_id"),
        call_index=call_index,
        model_ref=exchange.get("model_ref"),
        dialect=exchange.get("dialect"),
        status_code=exchange.get("status_code"),
        started_at=exchange.get("started_at"),
        completed_at=exchange.get("completed_at"),
        request=exchange.get("request"),
        response=response or None,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
        cache_hit=cache_hit,
        cached_tokens=cached_tokens,
        error_category=exchange.get("error_category"),
        latency_total_ms=exchange.get("latency_ms"),
        latency_ttft_ms=exchange.get("latency_ttft_ms"),
        **tokens,
    )


def read_model_call_records(store: CaptureStore, rollout_id: str) -> list[ModelCallRecord]:
    """Read captured exchanges in durable append order."""
    return [
        build_model_call_record(exchange, call_index=index) for index, exchange in enumerate(store.read(rollout_id))
    ]


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
_ROLLOUT_PATH_RE = re.compile(rf"^/{re.escape(ROLLOUT_PATH_PREFIX)}/(?P<rollout_id>[^/]+)(?P<rest>/.*)$")


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
    model_server_name: Optional[str],
    request_bytes: bytes,
    *,
    rollout_id: str,
    model_call_id: str,
    started_at: float,
    completed_at: float,
    response_body: Any,
    status_code: Optional[int],
    error_category: Optional[str],
    latency_ms: float,
    ttft_ms: Optional[float] = None,
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
        if request_raw is not None:
            exchange["request_raw"] = request_raw
        store.record(rollout_id, exchange)
    except Exception:
        logger.warning("Model-call capture failed for one %s call.", dialect, exc_info=True)


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
        model_call_id = uuid4().hex
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
                if chunk and state["ttft_ms"] is None:
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

        try:
            await self._app(scope, _receive, _send)
        except Exception as exc:
            completed_at = time.time()
            # Offload the blocking write+fsync so it never stalls the event loop.
            try:
                await asyncio.to_thread(
                    _record,
                    self._store,
                    dialect,
                    self._model_server_name,
                    bytes(request_body),
                    rollout_id=rollout_id,
                    model_call_id=model_call_id,
                    started_at=started_at,
                    completed_at=completed_at,
                    response_body=None,
                    status_code=None,
                    error_category=_classify_exception(exc),
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    ttft_ms=state["ttft_ms"],
                )
            except Exception:
                logger.warning("Model-call capture finalization failed.", exc_info=True)
            finally:
                await _flush_deferred_response()
            raise

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
            _record(
                store,
                dialect,
                model_server_name,
                request_bytes,
                rollout_id=rollout_id,
                model_call_id=model_call_id,
                started_at=started_at,
                completed_at=completed_at,
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


def install_model_call_capture(
    app: Any, config: ModelCallCaptureConfig, *, model_server_name: Optional[str] = None
) -> None:
    """Install model-call capture middleware.

    Always installed so the ``/ng-rollout/<id>`` correlation prefix is stripped before routing
    regardless of whether capture is enabled (otherwise a default ``gym eval`` would 404 on every
    prefixed model call). When capture is enabled the middleware additionally records each observed
    call's request + response into a rollout-keyed CaptureStore while forwarding bytes downstream
    unchanged (non-terminal SSE chunks are forwarded as they arrive; the terminal event follows the
    durable capture write).
    """
    app.add_middleware(
        _CaptureMiddleware,
        store=make_capture_store(config),
        model_server_name=model_server_name,
    )


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
    """Remove stale per-rollout capture files for these records before dispatch.

    Capture files are keyed by a deterministic rollout id (task-rollout-attempt), so without this a
    fresh run or a kill-shaped retry would append onto the previous attempt's capture for the same
    id. The caller passes only rows about to be dispatched, after assigning any retry suffix.
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
