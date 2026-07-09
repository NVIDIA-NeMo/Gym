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
"""Rollout-keyed capture storage and observability records for model calls."""

from __future__ import annotations

import fcntl
import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


def _sanitize(rollout_id: str) -> str:
    cleaned = "".join(c for c in rollout_id if c.isalnum() or c in ("-", "_", "."))
    return cleaned or "rollout"


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
        return self._root / f"{_sanitize(rollout_id)}.capture.jsonl"

    def record(self, rollout_id: str, exchange: dict[str, Any]) -> None:
        """Append one exchange and fsync (durable across a killed box).

        ``flock`` serializes appends across worker processes (a model server may run with
        ``num_workers > 1``, where the in-process lock can't coordinate); the in-process lock
        serializes threads. This does blocking file IO + fsync, so callers run it off the event
        loop (the capture middleware offloads it via ``asyncio.to_thread``).
        """
        line = json.dumps(exchange, default=str, ensure_ascii=False)
        path = self.path_for(rollout_id)
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.write(line + "\n")
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
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    exchanges.append(json.loads(stripped))
                except json.JSONDecodeError:
                    continue
        return exchanges


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

    # Durable append order, not a causal or semantic order for concurrent calls.
    call_index: int
    model_server: Optional[str] = None
    dialect: Optional[str] = None
    status_code: Optional[int] = None

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
        call_index=call_index,
        model_server=exchange.get("model_server"),
        dialect=exchange.get("dialect"),
        status_code=exchange.get("status_code"),
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
