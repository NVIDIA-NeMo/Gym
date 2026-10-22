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

"""Record every model call a Gym model server serves, for debugging.

```yaml
model_call_debug_trace:
  enabled: true
  dir: /absolute/path/model-call-debug   # Shared by every model server and worker process.
  request_token_data: true               # Ask the engine for token data the server would not request.
```

Each serving process appends one JSON line per model call to ``<dir>/<model_server>.<pid>.jsonl``.
A line holds the request the client sent, the response it received, and the call's prompt token ids,
generated token ids, and generation log probabilities from the first source that has them:

1. ``engine_staged``: the arrays an inference worker staged for external token capture.
   The worker echoes them under ``ng_debug_tokens`` only when the request asks for it.
2. ``served_response``: the token fields on the response returned to the client.
3. ``engine_response``: the token fields on the raw engine response, read before the server removes them.

``request_token_data`` adds the request fields that make the engine return token data.
The model server removes the extra data again, so the client receives the same response either way.
Tracing never fails or delays a model call: one background thread per process writes the lines.
Lines are flushed but not fsynced.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import orjson
from fastapi import Request
from pydantic import BaseModel, ConfigDict, model_validator

from nemo_gym.rollout_correlation import current_rollout_id
from nemo_gym.token_id_capture.adapters.vllm import extract_generation_token_info
from nemo_gym.token_id_capture.records import TOKEN_FIELDS, extract_token_fields
from nemo_gym.token_id_capture.sink import NG_CAPTURE_FIELD, NG_DEBUG_TOKENS_FIELD, current_capture_context


logger = logging.getLogger(__name__)

TRACE_SCHEMA_VERSION = 1
# Request-state key under which the capture middleware publishes the call's ``model_call_id``.
MODEL_CALL_ID_STATE_KEY = "nemo_gym_model_call_id"
_TOKEN_KEYS = ("prompt_token_ids", "generation_token_ids", "generation_log_probs")


class ModelCallDebugTraceSettings(BaseModel):
    """The ``model_call_debug_trace`` global-config block."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # Absolute directory that receives one JSONL file per model server process.
    dir: Path | None = None
    # Add engine request fields that return token data, and remove that data before responding.
    request_token_data: bool = True

    @model_validator(mode="after")
    def _require_absolute_dir(self) -> "ModelCallDebugTraceSettings":
        if self.enabled and (self.dir is None or not self.dir.is_absolute()):
            raise ValueError("model_call_debug_trace.enabled requires an absolute model_call_debug_trace.dir")
        return self


class _GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_call_debug_trace: ModelCallDebugTraceSettings = ModelCallDebugTraceSettings()


@dataclass
class CallTrace:
    """Debug evidence collected while one model call is served."""

    request_token_data: bool
    started_at: float = field(default_factory=time.time)
    started_perf: float = field(default_factory=time.perf_counter)
    response: Any = None
    engine_tokens: dict[str, Any] | None = None
    engine_tokens_source: str | None = None
    token_errors: list[str] = field(default_factory=list)
    # Engine request fields this trace added; their response data is removed before responding.
    added_request_fields: set[str] = field(default_factory=set)


_CALL_TRACE: ContextVar[CallTrace | None] = ContextVar("nemo_gym_model_call_debug_trace", default=None)


def current_call_trace() -> CallTrace | None:
    """Return the trace of the model call being served, or ``None`` when tracing is off."""
    return _CALL_TRACE.get()


def request_engine_token_data(request_payload: dict[str, Any], *, megatron_engine: bool) -> None:
    """Ask the engine to return the call's token data, when the active trace requests it.

    A call admitted for external capture asks the worker to echo the arrays it stages.
    Any other call requests the engine's ordinary token fields.
    """
    trace = _CALL_TRACE.get()
    if trace is None or not trace.request_token_data or request_payload.get("n") not in (None, 1):
        return
    if NG_CAPTURE_FIELD in request_payload:
        request_payload[NG_DEBUG_TOKENS_FIELD] = True
        trace.added_request_fields.add(NG_DEBUG_TOKENS_FIELD)
        return
    offload_params = request_payload.get("offload_params")
    if isinstance(offload_params, dict) and NG_CAPTURE_FIELD in offload_params:
        # Megatron Inference hands offload_params to the stager, whose metadata reaches the response.
        offload_params[NG_DEBUG_TOKENS_FIELD] = True
        trace.added_request_fields.add(NG_DEBUG_TOKENS_FIELD)
        return
    if not request_payload.get("logprobs"):
        # top_logprobs=0 returns only the selected token's log probability.
        request_payload.update(logprobs=True, top_logprobs=0, return_tokens_as_token_ids=True)
        trace.added_request_fields.add("logprobs")
    token_ids_field = "return_tokenized_data" if megatron_engine else "return_token_ids"
    if not request_payload.get(token_ids_field):
        request_payload[token_ids_field] = True
        trace.added_request_fields.add(token_ids_field)


def note_engine_response(response_payload: dict[str, Any]) -> None:
    """Record the token data on a raw engine response, then remove what tracing added.

    Call this before the model server converts or strips the engine response.
    """
    trace = _CALL_TRACE.get()
    if trace is None:
        return
    staged = response_payload.pop(NG_DEBUG_TOKENS_FIELD, None)
    try:
        if isinstance(staged, Mapping):
            if staged.get("error"):
                trace.token_errors.append(f"engine_staged: {staged['error']}")
            tokens, source = _token_bundle(staged), "engine_staged"
        else:
            tokens, source = _engine_response_tokens(response_payload), "engine_response"
    except (KeyError, TypeError, ValueError) as error:
        trace.token_errors.append(f"engine_response: {type(error).__name__}: {error}")
    else:
        if tokens is not None:
            trace.engine_tokens, trace.engine_tokens_source = tokens, source
    _remove_added_token_data(response_payload, trace.added_request_fields)


def _token_bundle(source: Mapping[str, Any]) -> dict[str, Any] | None:
    bundle = {key: source.get(key) for key in _TOKEN_KEYS}
    return bundle if any(value is not None for value in bundle.values()) else None


def _engine_response_tokens(response_payload: dict[str, Any]) -> dict[str, Any] | None:
    """Read token data from a Chat Completions engine response in any supported shape."""
    choices = response_payload.get("choices") or []
    if len(choices) != 1 or not isinstance(choices[0], dict):
        return None
    choice = choices[0]
    message = choice.get("message") or {}
    bundle = {
        "prompt_token_ids": message.get("prompt_token_ids") or response_payload.get("prompt_token_ids"),
        "generation_token_ids": message.get("generation_token_ids") or choice.get("token_ids"),
        "generation_log_probs": message.get("generation_log_probs"),
    }
    if bundle["generation_log_probs"] is None and (choice.get("logprobs") or {}).get("content") is not None:
        bundle["generation_token_ids"], bundle["generation_log_probs"] = extract_generation_token_info(choice)
    return _token_bundle(bundle)


def _remove_added_token_data(response_payload: dict[str, Any], added: set[str]) -> None:
    """Remove the response data that only exists because tracing changed the request."""
    if "return_token_ids" in added:
        response_payload.pop("prompt_token_ids", None)
    for choice in response_payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        if "return_token_ids" in added:
            choice.pop("token_ids", None)
        if "logprobs" in added:
            choice.pop("logprobs", None)
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        if "logprobs" in added:
            # The NeMo RL vLLM server attaches these whenever token-id logprobs are requested.
            for key in _TOKEN_KEYS:
                message.pop(key, None)
        if "return_tokenized_data" in added:
            message.pop("prompt_token_ids", None)
            message.pop("generation_token_ids", None)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return deepcopy(value)


def _without_token_fields(payload: Any) -> Any:
    """Drop token arrays from a served-response copy; the record keeps them once, under ``tokens``."""
    if not isinstance(payload, dict):
        return payload
    items = list(payload.get("output") or [])
    items += [choice.get("message") for choice in payload.get("choices") or [] if isinstance(choice, dict)]
    for item in items:
        if isinstance(item, dict):
            for key in TOKEN_FIELDS:
                item.pop(key, None)
    return payload


def _state_model_call_id(request: Request) -> str | None:
    """Return the id the capture middleware assigned to this call, if it assigned one."""
    model_call_id = (request.scope.get("state") or {}).get(MODEL_CALL_ID_STATE_KEY)
    return model_call_id if isinstance(model_call_id, str) else None


class ModelCallDebugTrace:
    """Write one debug record per model call served by this process."""

    def __init__(self, settings: ModelCallDebugTraceSettings, *, server_name: str) -> None:
        if settings.dir is None:
            raise ValueError("model_call_debug_trace requires a directory")
        self._directory = settings.dir
        self._server_name = server_name
        self._request_token_data = settings.request_token_data
        # One writer thread keeps this process's lines whole and in call-completion order.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-call-debug-trace")
        self._directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_global_config(cls, global_config_dict: Any, *, server_name: str) -> "ModelCallDebugTrace | None":
        """Build the trace writer for this process, or return ``None`` when tracing is disabled."""
        settings = _GlobalConfig.model_validate(global_config_dict or {}).model_call_debug_trace
        if not settings.enabled:
            return None
        return cls(settings, server_name=server_name)

    @asynccontextmanager
    async def call(self, request: Request, params: BaseModel) -> AsyncIterator[CallTrace]:
        """Trace one model call; a nested call reuses the outer call's trace."""
        outer = _CALL_TRACE.get()
        if outer is not None:
            yield outer
            return
        trace = CallTrace(request_token_data=self._request_token_data)
        reset_token = _CALL_TRACE.set(trace)
        error: BaseException | None = None
        try:
            yield trace
        except BaseException as exc:
            error = exc
            raise
        finally:
            _CALL_TRACE.reset(reset_token)
            try:
                self._submit(trace, request, params, error)
            except Exception:
                logger.warning("Could not record a model-call debug trace.", exc_info=True)

    def _submit(self, trace: CallTrace, request: Request, params: BaseModel, error: BaseException | None) -> None:
        response = _jsonable(trace.response) if trace.response is not None else None
        tokens, tokens_source = trace.engine_tokens, trace.engine_tokens_source
        if tokens_source != "engine_staged" and isinstance(response, dict):
            try:
                served = extract_token_fields(response)
            except ValueError as served_error:
                trace.token_errors.append(f"served_response: {served_error}")
            else:
                if served is not None:
                    tokens, tokens_source = _token_bundle(served), "served_response"
        response = _without_token_fields(response)
        context = current_capture_context()
        completed_at = time.time()
        record = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "trace_id": uuid4().hex,
            "rollout_id": current_rollout_id() or (context.rollout_id if context is not None else None),
            "model_call_id": context.model_call_id if context is not None else _state_model_call_id(request),
            "model_server": self._server_name,
            "path": request.url.path,
            "pid": os.getpid(),
            "started_at": trace.started_at,
            "completed_at": completed_at,
            "latency_ms": round((time.perf_counter() - trace.started_perf) * 1000.0, 2),
            "external_staging": bool(context is not None and context.external_staging),
            "commit_coords": context.external_commit_coords if context is not None else None,
            "request": params.model_dump(mode="json", exclude_unset=True),
            "response": response,
            "tokens": tokens,
            "tokens_source": tokens_source,
            "token_errors": trace.token_errors,
            "error": {"type": type(error).__name__, "message": str(error)} if error is not None else None,
        }
        self._executor.submit(self._append, record)

    def _append(self, record: dict[str, Any]) -> None:
        path = self._directory / f"{self._server_name}.{os.getpid()}.jsonl"
        try:
            line = orjson.dumps(record, default=str) + b"\n"
            with path.open("ab") as handle:
                handle.write(line)
        except (OSError, TypeError):
            logger.warning("Could not append a model-call debug trace to %s.", path, exc_info=True)

    def flush(self) -> None:
        """Wait for every submitted record to be written."""
        self._executor.submit(lambda: None).result()


@asynccontextmanager
async def trace_model_call(
    tracer: ModelCallDebugTrace | None, request: Request, params: BaseModel
) -> AsyncIterator[CallTrace | None]:
    """Trace one model call when ``tracer`` is configured; otherwise yield ``None``."""
    if tracer is None:
        yield None
        return
    async with tracer.call(request, params) as trace:
        yield trace
