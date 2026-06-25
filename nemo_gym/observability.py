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
"""Per-rollout trajectory capture for model servers.

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

from nemo_gym.server_utils import ROLLOUT_HEADER, ROLLOUT_PATH_PREFIX, rollout_id_from_run_body
from nemo_gym.trajectory_capture import CaptureStore, aggregate_rollout_metrics, assemble_step_records


logger = logging.getLogger(__name__)

# Grouping headers. ROLLOUT_HEADER is the shared protocol (server_utils); trial/turn are local.
TRIAL_HEADER = "x-nemo-gym-trial-index"
TURN_HEADER = "x-nemo-gym-turn-index"

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


def _scope_header_int(scope: dict[str, Any], name: str) -> Optional[int]:
    value = _scope_header(scope, name)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _headers_content_type(headers: list) -> bytes:
    for key, value in headers:
        if key.lower() == b"content-type":
            return value
    return b""


# Consumer side of the URL-prefix protocol: strip /ng-rollout/<id> before routing, key capture by
# <id>. The constant + producer (apply_rollout_prefix) are in server_utils.
_ROLLOUT_PATH_RE = re.compile(rf"^/{re.escape(ROLLOUT_PATH_PREFIX)}/(?P<rollout_id>[^/]+)(?P<rest>/.*)$")


# --- Compact per-call telemetry (utility; the canonical record is the full exchange) ---
def _usage(usage: Any) -> Optional[dict[str, Any]]:
    """Normalize token usage across Responses, Chat Completions, and Anthropic Messages."""
    if not usage:
        return None
    tokens_in = usage.get("input_tokens")
    if tokens_in is None:
        tokens_in = usage.get("prompt_tokens")
    tokens_out = usage.get("output_tokens")
    if tokens_out is None:
        tokens_out = usage.get("completion_tokens")
    tokens_total = usage.get("total_tokens")
    if tokens_total is None and tokens_in is not None and tokens_out is not None:
        tokens_total = tokens_in + tokens_out
    details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    return {
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_total": tokens_total,
        "tokens_reasoning": details.get("reasoning_tokens"),
    }


def summarize_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Compact CLU telemetry for a response payload (token stats + tool/message/reasoning signals).

    Handles all three response shapes: Responses (``output``), Chat Completions
    (``choices``), and Anthropic Messages (``content``).
    """
    summary: dict[str, Any] = {
        "model": payload.get("model"),
        "usage": _usage(payload.get("usage")),
        "num_tool_calls": 0,
        "tool_names": [],
        "num_messages": 0,
        "has_reasoning": False,
    }

    output = payload.get("output")
    if output is not None:  # Responses
        tool_calls = [item for item in output if item.get("type") == "function_call"]
        summary.update(
            num_tool_calls=len(tool_calls),
            tool_names=[call.get("name") for call in tool_calls],
            num_messages=sum(item.get("type") == "message" for item in output),
            has_reasoning=any(item.get("type") == "reasoning" for item in output),
        )
        return summary

    choices = payload.get("choices")
    if choices is not None:  # Chat Completions
        messages = [c.get("message") for c in choices if isinstance(c, dict) and c.get("message")]
        tool_calls = [tc for message in messages for tc in (message.get("tool_calls") or [])]
        summary.update(
            num_tool_calls=len(tool_calls),
            tool_names=[(tc.get("function") or {}).get("name") for tc in tool_calls],
            num_messages=len(messages),
            has_reasoning=any(message.get("reasoning_content") for message in messages),
        )
        return summary

    content = payload.get("content")
    if isinstance(content, list):  # Anthropic Messages
        tool_calls = [block for block in content if block.get("type") == "tool_use"]
        text_blocks = [block for block in content if block.get("type") == "text"]
        thinking = [block for block in content if block.get("type") in ("thinking", "redacted_thinking")]
        summary.update(
            num_tool_calls=len(tool_calls),
            tool_names=[block.get("name") for block in tool_calls],
            num_messages=1 if text_blocks else 0,
            has_reasoning=bool(thinking),
        )
        return summary

    return summary


# --- Full per-rollout exchange capture (the canonical trajectory source) ---
def _default_capture_dir(server_name: str) -> str:
    env_dir = os.environ.get("NEMO_GYM_TRAJECTORY_DIR")
    if env_dir:
        return env_dir
    return str(Path(tempfile.gettempdir()) / "nemo_gym_trajectories" / server_name)


def make_capture_store(config: Any) -> Optional[CaptureStore]:
    """Build a CaptureStore when observability is enabled; otherwise None."""
    if not getattr(config, "observability_enabled", False):
        return None
    root = getattr(config, "trajectory_capture_dir", None) or _default_capture_dir(
        getattr(config, "name", None) or "model_server"
    )
    try:
        return CaptureStore(root)
    except Exception:
        logger.warning("Could not initialize trajectory capture at %s; disabling it.", root, exc_info=True)
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


def _record(
    store: CaptureStore,
    scope: dict[str, Any],
    dialect: str,
    config: Any,
    request_bytes: bytes,
    *,
    rollout_id: str,
    response_body: Any,
    status_code: Optional[int],
    error_category: Optional[str],
    latency_ms: float,
) -> None:
    """Append one exchange (success or failure). Best-effort: never raises."""
    try:
        store.record(
            rollout_id,
            {
                "dialect": dialect,
                "model_server": getattr(config, "name", None),
                "trial_index": _scope_header_int(scope, TRIAL_HEADER),
                "turn_index": _scope_header_int(scope, TURN_HEADER),
                "latency_ms": round(latency_ms, 2),
                "status_code": status_code,
                "error_category": error_category,
                "request": json.loads(request_bytes) if request_bytes else None,
                "response": response_body,
            },
        )
    except Exception:
        logger.warning("Trajectory capture failed for one %s call.", dialect, exc_info=True)


class _CaptureMiddleware:
    """Pure-ASGI per-rollout capture.

    Buffers the request body and a copy of the response while forwarding both downstream
    unchanged, so it composes with streaming (SSE) responses -- it never consumes or rewraps
    the stream. An optional ``/ng-rollout/<id>`` path prefix is stripped before routing and used
    as the capture key (an explicit header wins). Streaming responses are forwarded but their
    body is not buffered.
    """

    def __init__(self, app: Any, *, store: CaptureStore, config: Any) -> None:
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

        state: dict[str, Any] = {"status": None, "streaming": False, "body": bytearray()}
        start = time.perf_counter()

        async def _send(message: dict[str, Any]) -> None:
            message_type = message.get("type")
            if message_type == "http.response.start":
                state["status"] = message.get("status")
                content_type = _headers_content_type(message.get("headers") or [])
                state["streaming"] = content_type.startswith(b"text/event-stream")
            elif message_type == "http.response.body" and not state["streaming"]:
                state["body"].extend(message.get("body", b"") or b"")
            await send(message)  # forward unchanged -> streaming is preserved

        try:
            await self._app(scope, _receive, _send)
        except Exception as exc:
            _record(
                self._store, scope, dialect, self._config, bytes(request_body),
                rollout_id=rollout_id,
                response_body=None,
                status_code=None,
                error_category=_classify_exception(exc),
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
            raise

        latency_ms = (time.perf_counter() - start) * 1000.0
        response_body = None
        if not state["streaming"] and state["body"]:
            try:
                response_body = json.loads(bytes(state["body"]))
            except Exception:
                response_body = None
        status = state["status"]
        _record(
            self._store, scope, dialect, self._config, bytes(request_body),
            rollout_id=rollout_id,
            response_body=response_body,
            status_code=status,
            error_category=_classify_status(status) if status is not None else None,
            latency_ms=latency_ms,
        )


def install_trajectory_capture(app: Any, config: Any) -> None:
    """Install the per-rollout exchange-capture middleware (no-op when capture is disabled).

    A pure-ASGI middleware that records each observed call's request + response into a
    rollout-keyed CaptureStore while forwarding bytes downstream unchanged, so it composes with
    streaming responses. Streaming (SSE) responses are forwarded but their body is not buffered.
    """
    store = make_capture_store(config)
    if store is None:
        return
    app.add_middleware(_CaptureMiddleware, store=store, config=config)


# --- Consumer read: fold per-rollout capture into the rollout record (uniform across agents) ---
def capture_dirs_from_config(global_config_dict: Any, env: Optional[dict[str, str]] = None) -> list[Path]:
    """Candidate directories to read per-rollout captures from.

    Collects ``$NEMO_GYM_TRAJECTORY_DIR`` (the shared sink) plus the resolved directory of every
    observability-enabled model server in the global config (its ``trajectory_capture_dir``, else the
    shared dir, else the per-server temp default). Deduped; existing dirs only. Best-effort.
    """
    environ = env if env is not None else os.environ
    dirs: list[Path] = []
    shared = environ.get("NEMO_GYM_TRAJECTORY_DIR")
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

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("observability_enabled"):
                resolved = node.get("trajectory_capture_dir") or shared or _default_capture_dir(
                    node.get("name") or "model_server"
                )
                dirs.append(Path(resolved))
            for value in node.values():
                _walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                _walk(value)

    try:
        _walk(global_config_dict)
    except Exception:
        logger.debug("Could not resolve capture dirs from the global config.", exc_info=True)

    seen: set[Path] = set()
    unique: list[Path] = []
    for directory in dirs:
        if directory not in seen and directory.exists():
            seen.add(directory)
            unique.append(directory)
    return unique


def _store_for_rollout(rollout_id: str, capture_dirs: list[Path]) -> Optional[CaptureStore]:
    for directory in capture_dirs:
        store = CaptureStore(directory)
        if store.path_for(rollout_id).exists():
            return store
    return None


def merge_capture_into_record(
    record: dict[str, Any], capture_dirs: list[Path], *, include_payloads: bool = False
) -> dict[str, Any]:
    """Attach a rollout's captured model-call trajectory to its rollout record, in place.

    Keyed by the rollout id derived from the record's task/rollout/attempt indices, so the attached
    shape is identical for every agent harness. Adds
    ``ng_trajectory_capture = {rollout_id, metrics, steps}`` where ``steps`` are typed StepRecords
    (raw request/response excluded unless ``include_payloads`` -- they remain in the capture store).
    No-op when no capture exists. The harness output + reward on the record are never modified: the
    capture is authoritative for per-step model-call stats, the harness for reward.
    """
    if not capture_dirs:
        return record
    rollout_id = rollout_id_from_run_body(record)
    if rollout_id is None:
        return record
    store = _store_for_rollout(rollout_id, capture_dirs)
    if store is None:
        return record
    steps = assemble_step_records(store, rollout_id)
    if not steps:
        return record
    exclude = None if include_payloads else {"request", "response"}
    record["ng_trajectory_capture"] = {
        "rollout_id": rollout_id,
        "metrics": aggregate_rollout_metrics(store, rollout_id),
        "steps": [step.model_dump(exclude=exclude) for step in steps],
    }
    return record
