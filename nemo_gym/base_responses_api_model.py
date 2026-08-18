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
import time
from abc import abstractmethod
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Mapping, Optional
from uuid import uuid4

import orjson
from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError, model_validator

from nemo_gym.anthropic_converter import AnthropicConverter
from nemo_gym.chat_streaming import sanitize_streaming_chat_body, synthesize_chat_completion_sse
from nemo_gym.config_types import ROLLOUT_PATH_PREFIX, TOKEN_CAPTURE_PATH_SEGMENT, ModelServerRef
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
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.rollout_observability import AgentObservationBundle, ObservationGap, join_model_call_observations
from nemo_gym.server_utils import (
    BaseRunServerInstanceConfig,
    BaseServer,
    SimpleServer,
)
from nemo_gym.token_id_capture import (
    CaptureContext,
    capture_tokens,
    installed_lineage_store,
    installed_token_sink,
    reset_token_sink,
    resolve_parent,
    set_token_sink,
)

# The store factory needs Gym's server stack.
# The leaf package does not re-export it.
from nemo_gym.token_id_capture.config import token_id_capture_config
from nemo_gym.token_id_capture.lineage import FileLineageStore, InMemoryLineageStore
from nemo_gym.token_id_capture.store import make_token_store


logger = logging.getLogger(__name__)


# Stateless; shared by every model server's default /v1/messages handler.
_ANTHROPIC_CONVERTER = AnthropicConverter()


def _request_messages(body: Any) -> list[dict]:
    """Return the conversation carried by any supported dialect.

    Lineage uses model-authored turns to identify the parent call.
    Chat and Anthropic use ``messages``.
    Responses uses ``input``.
    The request envelope is prepended as a pseudo-turn because it also shapes the prompt.
    """
    if body is None:
        return []
    getter = body.get if isinstance(body, dict) else lambda key, default=None: getattr(body, key, default)
    messages = getter("messages", None)
    if isinstance(messages, list):
        turns = [m if isinstance(m, dict) else m.model_dump() for m in messages if m is not None]
    else:
        items = getter("input", None)
        turns = (
            [i if isinstance(i, dict) else i.model_dump() for i in items if i is not None]
            if isinstance(items, list)
            else []
        )
    return _request_envelope(getter) + turns


# This role cannot collide with a dialect role.
# It is not assistant-authored and does not affect the lookup fingerprint.
_ENVELOPE_ROLE = "_ng_request_envelope"


def _request_envelope(getter: Any) -> list[dict]:
    """Return prompt-shaping request fields that are not turns.

    Instructions and tools can be siblings of the message list.
    The chat template renders both into the prompt.
    Including them prevents prefix reuse across different request envelopes.
    """
    instructions = getter("instructions", None)
    tools = getter("tools", None)
    if not instructions and not tools:
        return []
    envelope = {"instructions": _plain(instructions), "tools": _plain(tools)}
    return [{"role": _ENVELOPE_ROLE, "content": json.dumps(envelope, sort_keys=True, default=str)}]


def _plain(value: Any) -> Any:
    """Reduce a request field to plain data so equal schemas serialize equally.

    Handlers can expose tools as dictionaries or Pydantic models.
    Normalization keeps an unchanged schema stable across handlers.
    """
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


class BaseResponsesAPIModelConfig(BaseRunServerInstanceConfig):
    pass


class BaseResponsesAPIModel(BaseServer):
    config: BaseResponsesAPIModelConfig


class SimpleResponsesAPIModel(BaseResponsesAPIModel, SimpleServer):
    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        self.setup_session_middleware(app)
        capture_config = ModelCallCaptureConfig.model_validate(self.server_client.global_config_dict)
        install_model_call_capture(
            app,
            capture_config,
            model_server_name=self.config.name,
            global_config_dict=self.server_client.global_config_dict,
            num_workers=self.config.num_workers,
        )

        app.post("/v1/chat/completions")(self.chat_completions_dispatch)

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

    async def chat_completions_dispatch(self, request: Request, body: dict = Body()):
        """Default ``/v1/chat/completions`` entrypoint shared by every Gym model server.

        A non-streaming request validates strictly against
        ``NeMoGymChatCompletionCreateParamsNonStreaming`` and delegates to this server's own
        ``chat_completions()``, preserving the historical non-streaming behavior (including the
        standard 422 shape). When the client sends ``stream: true`` (blackbox
        Chat-Completions-over-SSE harnesses like the OpenClaw agent always do), the request is
        sanitized onto that same strict model (drop ``stream``/``stream_options``; see
        ``nemo_gym.chat_streaming``), validated identically, delegated to the same
        ``chat_completions()``, and the complete response is buffered and re-emitted as a
        synthesized ``chat.completion.chunk`` SSE stream. This is buffer-then-replay, not
        token-by-token streaming.

        Only a genuine boolean ``stream: true`` takes the streaming path; any other value
        (e.g. ``"false"`` or ``1``) stays on the strict non-streaming path, which rejects the
        malformed ``stream`` with the same 422 as before.
        """
        if body.get("stream") is not True:
            params = _validate_chat_params(body)
            return await self._invoke_chat_completions(request, params)

        cleaned, include_usage = sanitize_streaming_chat_body(body)
        params = _validate_chat_params(cleaned)
        completion = await self._invoke_chat_completions(request, params)
        completion_json = completion.model_dump(mode="json") if isinstance(completion, BaseModel) else dict(completion)
        return StreamingResponse(
            synthesize_chat_completion_sse(completion_json, include_usage=include_usage),
            media_type="text/event-stream",
        )

    async def _invoke_chat_completions(
        self, request: Request, params: NeMoGymChatCompletionCreateParamsNonStreaming
    ) -> NeMoGymChatCompletion:
        # chat_completions() signatures vary across servers: some take a leading `request`, some
        # only `body`. Dispatch on whichever this server declares so the shared dispatch works for
        # all of them.
        # Resolve the parent from the received request before dispatch.
        # Exact prefix supply and capture share this decision.
        await resolve_parent(_request_messages(params))
        if "request" in inspect.signature(self.chat_completions).parameters:
            completion = await self.chat_completions(request=request, body=params)
        else:
            completion = await self.chat_completions(body=params)
        await capture_tokens(
            completion,
            request_messages=_request_messages(params),
        )
        return completion

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
        # Resolve the parent from the received request before dispatch.
        # Exact prefix supply and capture share this decision.
        await resolve_parent(_request_messages(params))
        if "request" in inspect.signature(self.responses).parameters:
            response = await self.responses(request=request, body=params)
        else:
            response = await self.responses(body=params)
        # Capture before streaming dispatch wraps the response.
        # Anthropic mapping drops the token fields.
        # The assembled response still carries them here for every dialect.
        await capture_tokens(
            response,
            request_messages=_request_messages(params),
        )
        return response


def _validate_responses_params(body: dict) -> NeMoGymResponseCreateParamsNonStreaming:
    """Validate a /v1/responses body dict, surfacing failures as FastAPI's standard 422."""
    try:
        return NeMoGymResponseCreateParamsNonStreaming.model_validate(body)
    except ValidationError as exc:
        raise RequestValidationError([{**error, "loc": ("body", *error["loc"])} for error in exc.errors()])


def _validate_chat_params(body: dict) -> NeMoGymChatCompletionCreateParamsNonStreaming:
    """Validate a /v1/chat/completions body dict, surfacing failures as FastAPI's standard 422.

    ``include_url=False`` mirrors FastAPI's native body validation, which strips the
    ``errors.pydantic.dev`` url from each detail entry, so the 422 body stays byte-for-byte
    identical to the previous typed-``Body()`` binding.
    """
    try:
        return NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(body)
    except ValidationError as exc:
        raise RequestValidationError(
            [{**error, "loc": ("body", *error["loc"])} for error in exc.errors(include_url=False)]
        )


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


# --- Observability records derived from captured exchanges ---


def _token_count(value: Any) -> Optional[int]:
    return value if type(value) is int and value >= 0 else None


def _usage_detail_token(
    usage: Mapping[str, Any], detail_groups: tuple[str, ...], field_names: tuple[str, ...]
) -> Optional[int]:
    """Return the first valid token count across equivalent provider detail shapes."""
    for group_name in detail_groups:
        details = usage.get(group_name)
        if not isinstance(details, Mapping):
            continue
        for field_name in field_names:
            value = _token_count(details.get(field_name))
            if value is not None:
                return value
    return None


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
    if not isinstance(usage, Mapping):
        return {
            "tokens_in": None,
            "tokens_out": None,
            "tokens_reasoning": None,
            "tokens_total": None,
            "cache_creation_tokens": None,
        }
    tokens_in = _token_count(usage.get("input_tokens"))
    if tokens_in is None:
        tokens_in = _token_count(usage.get("prompt_tokens"))
    tokens_out = _token_count(usage.get("output_tokens"))
    if tokens_out is None:
        tokens_out = _token_count(usage.get("completion_tokens"))
    # Anthropic-native shape: top-level cache_* keys mean input_tokens excludes cached tokens.
    cache_read = _token_count(usage.get("cache_read_input_tokens"))
    cache_creation = _token_count(usage.get("cache_creation_input_tokens"))
    if cache_read is not None or cache_creation is not None:
        cache_total = (cache_read or 0) + (cache_creation or 0)
        # Zero cache fields alone do not establish a missing prompt count.
        if tokens_in is not None or cache_total > 0:
            tokens_in = (tokens_in or 0) + cache_total
    tokens_total = _token_count(usage.get("total_tokens"))
    if tokens_total is None and tokens_in is not None and tokens_out is not None:
        tokens_total = tokens_in + tokens_out
    tokens_reasoning = _usage_detail_token(
        usage, ("output_tokens_details", "completion_tokens_details"), ("reasoning_tokens",)
    )
    if tokens_reasoning is None:
        tokens_reasoning = _token_count(usage.get("reasoning_output_tokens"))
    return {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_reasoning": tokens_reasoning,
        "tokens_total": tokens_total,
        "cache_creation_tokens": cache_creation,
    }


def _cache_signal(usage: Any) -> tuple[Optional[bool], Optional[int]]:
    """Cache hit/miss + cached-token count, from usage cache fields (OpenAI / Anthropic)."""
    if not isinstance(usage, Mapping):
        return None, None
    cached = _usage_detail_token(
        usage,
        ("prompt_tokens_details", "input_tokens_details"),
        ("cached_tokens", "cached_input_tokens"),
    )
    if cached is None:
        cached = _token_count(usage.get("cache_read_input_tokens"))  # Anthropic
    if cached is None:
        cached = _token_count(usage.get("cached_input_tokens"))
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
                    if isinstance(text, str) and text:
                        reasoning.append(text)
        return tool_calls, ("\n".join(reasoning) or None)

    choices = response.get("choices")
    if isinstance(choices, list):  # Chat Completions
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(message, Mapping):
                continue
            for tc in message.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                if not isinstance(fn, Mapping):
                    fn = {}
                tool_calls.append(
                    {"call_id": tc.get("id"), "name": fn.get("name"), "arguments": _as_arguments(fn.get("arguments"))}
                )
            # vLLM and newer OpenAI-compatible servers emit `reasoning`; `reasoning_content` is the
            # older field. Accept either (reasoning_content wins when both are present).
            reasoning_text = message.get("reasoning_content") or message.get("reasoning")
            if isinstance(reasoning_text, str) and reasoning_text:
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
            elif block.get("type") in ("thinking", "redacted_thinking") and isinstance(block.get("thinking"), str):
                reasoning.append(block["thinking"])
        return tool_calls, ("\n".join(reasoning) or None)

    return tool_calls, None


class ModelCallRecord(BaseModel):
    """Observability record derived from one captured model-server exchange."""

    # Unique server-generated identity for each persisted call.
    model_call_id: Optional[str] = None
    response_id: Optional[str] = None

    # Durable append order, not a causal or semantic order for concurrent calls.
    call_index: int
    model_ref: Optional[ModelServerRef] = None
    model: Optional[str] = None
    dialect: Optional[str] = None
    status_code: Optional[int] = None
    response_status: Optional[str] = None
    finish_reason: Optional[str] = None

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
    request_raw: Optional[str] = None
    response_raw: Optional[str] = None
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
    raw_response = exchange.get("response")
    response = raw_response if isinstance(raw_response, dict) else {}
    tokens = extract_token_stats(response.get("usage"))
    cache_hit, cached_tokens = _cache_signal(response.get("usage"))
    tool_calls, reasoning_content = _tool_calls_and_reasoning(response)
    raw_request = exchange.get("request")
    request = raw_request if isinstance(raw_request, dict) else {}
    choices = response.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    incomplete_details = response.get("incomplete_details")
    if not isinstance(incomplete_details, dict):
        incomplete_details = {}
    finish_reason = next(
        (
            value
            for value in (
                response.get("stop_reason"),
                first_choice.get("finish_reason"),
                incomplete_details.get("reason"),
            )
            if isinstance(value, str)
        ),
        None,
    )
    model = response.get("model") or request.get("model")
    return ModelCallRecord(
        model_call_id=exchange.get("model_call_id"),
        response_id=response.get("id") if isinstance(response.get("id"), str) else None,
        call_index=call_index,
        model_ref=exchange.get("model_ref"),
        model=model if isinstance(model, str) else None,
        dialect=exchange.get("dialect"),
        status_code=exchange.get("status_code"),
        response_status=response.get("status") if isinstance(response.get("status"), str) else None,
        finish_reason=finish_reason,
        started_at=exchange.get("started_at"),
        completed_at=exchange.get("completed_at"),
        request=raw_request if isinstance(raw_request, dict) else None,
        response=raw_response if isinstance(raw_response, dict) else None,
        request_raw=exchange.get("request_raw") if isinstance(exchange.get("request_raw"), str) else None,
        response_raw=exchange.get("response_raw") if isinstance(exchange.get("response_raw"), str) else None,
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


def read_available_model_call_records(store: CaptureStore, rollout_id: str) -> tuple[list[ModelCallRecord], int]:
    """Read valid call records and count damaged records."""
    exchanges, invalid_count = store.read_available(rollout_id)
    calls = []
    for index, exchange in exchanges:
        try:
            calls.append(build_model_call_record(exchange, call_index=index))
        except Exception:
            invalid_count += 1
    return calls, invalid_count


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
        "cached_tokens": _sum("cached_tokens"),
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
        if response_raw is not None:
            exchange["response_raw"] = response_raw
        store.record(rollout_id, exchange)
    except Exception:
        logger.warning("Model-call capture failed for one %s call.", dialect, exc_info=True)
        try:
            store.mark_incomplete(rollout_id)
        except Exception:
            logger.warning("Could not mark rollout %s capture as incomplete.", rollout_id, exc_info=True)


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

    def __init__(
        self,
        app: Any,
        *,
        store: CaptureStore | None,
        model_server_name: str | None,
        token_store: Any = None,
        configured_sink: Any = None,
        lineage_store: Any = None,
        token_capture_enabled: bool = False,
    ) -> None:
        self._app = app
        self._store = store
        self._model_server_name = model_server_name
        # This store records training tokens for correlated training-capture calls.
        self._token_store = token_store
        # Built from token_id_capture.sink, once, in this process.
        self._configured_sink = configured_sink
        self._lineage_store = lineage_store
        # Capture may have no destination in this process.
        # A framework may stage records from its inference worker.
        # This process still resolves the capture identity.
        self._token_capture_enabled = token_capture_enabled

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        path = scope.get("path", "")
        rollout_from_path: Optional[str] = None
        token_capture_requested = False
        prefix_match = _ROLLOUT_PATH_RE.match(path)
        if prefix_match:
            rollout_from_path = prefix_match.group("rollout_id")
            token_capture_requested = prefix_match.group("token_capture") is not None
            path = prefix_match.group("rest")
            scope = {**scope, "path": path, "raw_path": path.encode("utf-8")}

        dialect = _OBSERVED_PATHS.get(path)

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
        if (self._store is None and not capture_wanted) or rollout_from_path is None or dialect is None:
            await self._app(scope, receive, send)
            return

        rollout_id = rollout_from_path
        model_call_id = uuid4().hex

        # Give the model server a token sink keyed to this call.
        # The sink records token ids from the complete response.
        # Middleware cannot recover token ids from SSE.
        # The context exists even without a local destination.
        # External staging uses the identity resolved here.
        sink_token = None
        if capture_wanted:
            sink_token = set_token_sink(
                CaptureContext(
                    rollout_id=rollout_id,
                    model_call_id=model_call_id,
                    token_sink=token_sink,
                    lineage_store=self._lineage_store,
                )
            )

        # Training-only capture has no evaluation record.
        # Forward without buffering while the sink is active.
        if self._store is None:
            try:
                await self._app(scope, receive, send)
            finally:
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
                )
            except Exception:
                logger.warning("Model-call capture finalization failed.", exc_info=True)
            finally:
                await _flush_deferred_response()
            raise
        finally:
            # The sink is only needed while the model server produces the response.
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
                response_raw=response_raw,
            )

        try:
            await asyncio.to_thread(_parse_and_record)
        except Exception:
            logger.warning("Model-call capture finalization failed.", exc_info=True)
        finally:
            await _flush_deferred_response()


def install_model_call_capture(
    app: Any,
    config: ModelCallCaptureConfig,
    *,
    model_server_name: str | None = None,
    global_config_dict: Any = None,
    num_workers: int | None = None,
) -> None:
    """Install model-call capture middleware.

    Always strip ``/ng-rollout/<id>/...`` before routing.
    Evaluation capture records requests and responses for that path.
    Non-terminal SSE chunks continue immediately.
    The terminal event follows the durable evaluation write.
    Training capture uses ``/ng-rollout/<id>/training-token-capture/...``.
    That path provides a request-scoped token sink.
    The model server records token ids from its complete response.
    Consumers access records through ``TokenSource.freeze``.
    There is no HTTP token reader.
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
    if lineage_store is None and capture_settings is not None and capture_settings.enabled:
        default_lineage = FileLineageStore(token_store.root) if token_store is not None else InMemoryLineageStore()
        lineage_store = default_lineage
    if capture_settings is not None and capture_settings.enabled and (num_workers or 1) > 1:
        if isinstance(lineage_store, InMemoryLineageStore):
            raise ValueError(
                "token_id_capture.lineage_store is required with num_workers > 1 "
                "when capture is not using Gym's shared file store"
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
        model_server_name=model_server_name,
        token_store=token_store,
        configured_sink=configured_sink,
        lineage_store=lineage_store,
        token_capture_enabled=capture_settings.enabled if capture_settings is not None else False,
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
        if store.path_for(rollout_id).exists() or store.is_incomplete(rollout_id):
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
                store.incomplete_path_for(rollout_id).unlink(missing_ok=True)


def merge_model_call_capture_into_record(
    record: dict[str, Any], capture_dirs: list[Path], *, include_payloads: bool = False
) -> dict[str, Any]:
    """Attach captured model-call observability data to a rollout record in place.

    Keyed by the rollout id derived from the record's task/rollout/attempt indices, so the attached
    shape is identical for every agent harness. Adds
    ``ng_model_call_capture = {rollout_id, metrics, calls}`` where ``calls`` are derived observability
    records. Raw request and response payloads remain in the capture store and are omitted from the
    attachment unless ``include_payloads`` is true. Capture/read/join failures are attached as
    ``gaps``. The harness output and reward are not modified.
    """
    if not capture_dirs:
        return record
    rollout_id = maybe_rollout_id_from_run_body(record)
    if rollout_id is None:
        return record
    gaps: list[ObservationGap] = []
    store = _store_for_rollout(rollout_id, capture_dirs)
    if store is None:
        calls = []
        gaps.append(ObservationGap(code="model_call_capture_no_records"))
    else:
        try:
            calls, invalid_count = read_available_model_call_records(store, rollout_id)
            if invalid_count:
                gaps.append(
                    ObservationGap(
                        code="model_call_capture_records_unreadable",
                        detail=f"count={invalid_count}",
                    )
                )
            if store.is_incomplete(rollout_id):
                gaps.append(ObservationGap(code="model_call_capture_incomplete"))
            elif not calls and not invalid_count:
                gaps.append(ObservationGap(code="model_call_capture_no_records"))
        except Exception:
            logger.warning("Could not read model-call capture for rollout %s.", rollout_id, exc_info=True)
            calls = []
            gaps.append(ObservationGap(code="model_call_capture_unreadable"))
    observations = record.get("ng_agent_observations")
    if observations is not None:
        try:
            bundle = AgentObservationBundle.model_validate(observations)
            if bundle.source == "claude_code":
                try:
                    from responses_api_agents.claude_code_agent.observability import (
                        associate_claude_code_compaction_calls,
                    )

                    bundle = associate_claude_code_compaction_calls(bundle, calls)
                except Exception:
                    logger.warning(
                        "Could not associate Claude Code compaction calls for rollout %s.",
                        rollout_id,
                        exc_info=True,
                    )
                    bundle.gaps.append(ObservationGap(code="compaction_model_call_join_failed"))
            bundle = join_model_call_observations(bundle, calls)
            record["ng_agent_observations"] = bundle.model_dump(mode="json")
        except Exception:
            logger.warning("Could not join agent observations for rollout %s.", rollout_id, exc_info=True)
            gaps.append(ObservationGap(code="agent_observation_join_failed"))
    exclude = None if include_payloads else {"request", "response", "request_raw", "response_raw"}
    capture = {
        "rollout_id": rollout_id,
        "metrics": aggregate_model_call_records(calls),
        "calls": [call.model_dump(exclude=exclude) for call in calls],
    }
    if gaps:
        capture["gaps"] = [gap.model_dump(mode="json", exclude_none=True) for gap in gaps]
    record["ng_model_call_capture"] = capture
    return record
