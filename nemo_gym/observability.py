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

Opt-in FastAPI middleware (off by default) that records every /v1/responses,
/v1/chat/completions, and /v1/messages exchange -- including failed calls -- into a
per-rollout CaptureStore. Best-effort; never alters the response. Correlation is
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

from fastapi.responses import Response

from nemo_gym.server_utils import ROLLOUT_HEADER, ROLLOUT_PATH_PREFIX
from nemo_gym.trajectory_capture import CaptureStore


logger = logging.getLogger(__name__)

# Grouping headers. ROLLOUT_HEADER is the shared protocol (server_utils); trial/turn are local.
TRIAL_HEADER = "x-nemo-gym-trial-index"
TURN_HEADER = "x-nemo-gym-turn-index"

_OBSERVED_PATHS = {
    "/v1/responses": "responses",
    "/v1/chat/completions": "chat",
    "/v1/messages": "messages",
}


def _header_int(request: Any, name: str) -> Optional[int]:
    value = request.headers.get(name)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


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
    request: Any,
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
                "trial_index": _header_int(request, TRIAL_HEADER),
                "turn_index": _header_int(request, TURN_HEADER),
                "latency_ms": round(latency_ms, 2),
                "status_code": status_code,
                "error_category": error_category,
                "request": json.loads(request_bytes) if request_bytes else None,
                "response": response_body,
            },
        )
    except Exception:
        logger.warning("Trajectory capture failed for one %s call.", dialect, exc_info=True)


def install_trajectory_capture(app: Any, config: Any) -> None:
    """Install the per-rollout exchange-capture middleware (no-op when capture is disabled).

    Records each observed call's request + response into a rollout-keyed CaptureStore.
    Signature-agnostic; never alters the response. Buffers and replays the request body.
    """
    store = make_capture_store(config)
    if store is None:
        return

    @app.middleware("http")
    async def _capture(request: Any, call_next: Any) -> Response:
        # Per-rollout URL prefix: strip it so /v1/... routes unchanged; use its id (header wins).
        path = request.url.path
        rollout_from_path: Optional[str] = None
        prefix_match = _ROLLOUT_PATH_RE.match(path)
        if prefix_match:
            rollout_from_path = prefix_match.group("rollout_id")
            path = prefix_match.group("rest")
            request.scope["path"] = path
            request.scope["raw_path"] = path.encode("utf-8")

        dialect = _OBSERVED_PATHS.get(path)
        if dialect is None:
            return await call_next(request)  # not observed (or a stripped non-/v1 path)

        rollout_id = request.headers.get(ROLLOUT_HEADER) or rollout_from_path or "rollout"
        request_bytes = await request.body()

        # Replay the buffered body so the route handler can still read it.
        async def _receive() -> dict[str, Any]:
            return {"type": "http.request", "body": request_bytes, "more_body": False}

        request._receive = _receive

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            # Capture the failed call, then propagate.
            _record(
                store,
                request,
                dialect,
                config,
                request_bytes,
                rollout_id=rollout_id,
                response_body=None,
                status_code=None,
                error_category=_classify_exception(exc),
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
            raise

        response_bytes = b"".join([chunk async for chunk in response.body_iterator])
        latency_ms = (time.perf_counter() - start) * 1000.0
        response_body = None
        if response_bytes:
            try:
                response_body = json.loads(response_bytes)
            except Exception:
                response_body = None
        _record(
            store,
            request,
            dialect,
            config,
            request_bytes,
            rollout_id=rollout_id,
            response_body=response_body,
            status_code=response.status_code,
            error_category=_classify_status(response.status_code),
            latency_ms=latency_ms,
        )

        headers = dict(response.headers)
        headers.pop("content-length", None)  # Response() recomputes it from the buffered body
        return Response(
            content=response_bytes,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
        )
