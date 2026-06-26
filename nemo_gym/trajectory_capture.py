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
"""Rollout-keyed capture store, eval trajectory assembly, and the #1483 StepRecord.

- CaptureStore: append-only per-rollout JSONL of model exchanges (durable, fsynced).
- assemble_trajectory / assemble_rollout: ordered NeMoGym output items for the eval view
  (content, tool calls, tool outputs); token-ids / log-probs are intentionally not surfaced
  (on-policy RL trajectory assembly is owned by the RL side and consumes this same capture).
- StepRecord / build_step_record / assemble_step_records / aggregate_rollout_metrics: the
  #1483 per-step stats contract and its per-rollout aggregates.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)


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


def _response_message(response: dict[str, Any]) -> dict[str, Any]:
    """First-choice assistant message (Chat Completions wire)."""
    if not isinstance(response, dict):
        return {}
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return {}
    message = choices[0].get("message") or choices[0].get("delta") or {}
    return message if isinstance(message, dict) else {}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            (part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")) for part in content
        )
    return "" if content is None else str(content)


def _assistant_message(index: int, text: str) -> NeMoGymResponseOutputMessage:
    """An assistant message for the eval-only trajectory view (no token-ids / log-probs)."""
    return NeMoGymResponseOutputMessage(
        id=f"msg-{index}",
        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
        role="assistant",
        status="completed",
        type="message",
    )


def assemble_trajectory(exchanges: list[dict[str, Any]], wire: str = "chat") -> list[Any]:
    """Reconstruct the ordered, eval-only NeMoGym output items (messages, tool calls, tool outputs).

    ``wire`` selects the captured body shape: ``"chat"`` (Chat Completions / Anthropic-translated)
    or ``"responses"`` (Responses API). Token-ids / log-probs are not surfaced here.
    """
    if wire == "responses":
        return _assemble_responses(exchanges)
    return _assemble_chat(exchanges)


def _assemble_chat(exchanges: list[dict[str, Any]]) -> list[Any]:
    """Chat-Completions wire. A turn's tool *results* arrive as ``role: tool`` messages in the
    *next* request; emit them before that turn's assistant message for natural ordering."""
    output: list[Any] = []
    seen_tool_call_ids: set[str] = set()
    assistant_index = 0

    for exchange in exchanges:
        request = exchange.get("request") or {}
        response = exchange.get("response") or {}

        for message in request.get("messages") or []:
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            call_id = message.get("tool_call_id") or ""
            if call_id and call_id in seen_tool_call_ids:
                continue
            if call_id:
                seen_tool_call_ids.add(call_id)
            output.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=_content_text(message.get("content")),
                    status="completed",
                )
            )

        message = _response_message(response)
        if not message:
            continue
        output.append(_assistant_message(assistant_index, _content_text(message.get("content"))))
        assistant_index += 1

        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            if not function:
                continue
            output.append(
                NeMoGymResponseFunctionToolCall(
                    arguments=function.get("arguments", ""),
                    call_id=tool_call.get("id", ""),
                    name=function.get("name", ""),
                    type="function_call",
                    id=tool_call.get("id"),
                    status="completed",
                )
            )

    return output


def _assemble_responses(exchanges: list[dict[str, Any]]) -> list[Any]:
    """Responses-API wire. Tool *results* arrive as ``function_call_output`` items in the *next*
    request's ``input``; emit them before that turn's response (mirrors the chat-wire assembler)."""
    output: list[Any] = []
    assistant_index = 0
    seen_output_call_ids: set[str] = set()
    for exchange in exchanges:
        request = exchange.get("request") or {}
        request_input = request.get("input")
        if isinstance(request_input, list):
            for item in request_input:
                if not isinstance(item, dict) or item.get("type") != "function_call_output":
                    continue
                call_id = item.get("call_id") or item.get("id") or ""
                if call_id and call_id in seen_output_call_ids:
                    continue
                if call_id:
                    seen_output_call_ids.add(call_id)
                output.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=call_id,
                        output=_content_text(item.get("output")),
                        status="completed",
                    )
                )

        response = exchange.get("response") or {}
        for item in response.get("output") or []:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "message" and item.get("role") == "assistant":
                text = "".join(
                    block.get("text", "")
                    for block in (item.get("content") or [])
                    if isinstance(block, dict) and block.get("type") == "output_text"
                )
                output.append(_assistant_message(assistant_index, text))
                assistant_index += 1
            elif kind in ("function_call", "tool_call"):
                output.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=item.get("arguments", "") or "",
                        call_id=item.get("call_id") or item.get("id") or "",
                        name=item.get("name", ""),
                        type="function_call",
                        id=item.get("id"),
                        status="completed",
                    )
                )
    return output


def assemble_rollout(store: CaptureStore, rollout_id: str) -> list[Any]:
    """Read a rollout's captured exchanges and assemble the full ordered trajectory.

    Picks the wire from the first exchange's recorded ``dialect`` (``responses`` for
    the Responses API, otherwise the chat assembler).
    """
    exchanges = store.read(rollout_id)
    wire = "responses" if exchanges and exchanges[0].get("dialect") == "responses" else "chat"
    return assemble_trajectory(exchanges, wire=wire)


# --- #1483 step contract: typed StepRecord + builders ---
def extract_token_stats(usage: Any) -> dict[str, Optional[int]]:
    """Normalize token totals across Responses, Chat Completions, and Anthropic Messages usage.

    For native Anthropic ``/v1/messages`` with prompt caching, ``input_tokens`` is only the uncached
    remainder, so cache-read + cache-creation tokens are folded into ``tokens_in`` to reflect the true
    prompt size (and cache-creation is surfaced separately as ``cache_creation_tokens``). OpenAI /
    Responses usage already includes cached tokens in ``input_tokens`` / ``prompt_tokens`` (where
    ``cached_tokens`` is a subset), so it is left untouched -- no double counting.
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
    if tokens_in is not None and (cache_read is not None or cache_creation is not None):
        tokens_in = tokens_in + (cache_read or 0) + (cache_creation or 0)
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
    if output is not None:  # Responses
        for item in output:
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
    if choices is not None:  # Chat Completions
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if not message:
                continue
            for tc in message.get("tool_calls") or []:
                fn = tc.get("function") or {}
                tool_calls.append(
                    {"call_id": tc.get("id"), "name": fn.get("name"), "arguments": _as_arguments(fn.get("arguments"))}
                )
            if message.get("reasoning_content"):
                reasoning.append(message["reasoning_content"])
        return tool_calls, ("\n".join(reasoning) or None)

    content = response.get("content")
    if isinstance(content, list):  # Anthropic Messages
        for block in content:
            if block.get("type") == "tool_use":
                tool_calls.append(
                    {"call_id": block.get("id"), "name": block.get("name"), "arguments": block.get("input") or {}}
                )
            elif block.get("type") in ("thinking", "redacted_thinking") and block.get("thinking"):
                reasoning.append(block["thinking"])
        return tool_calls, ("\n".join(reasoning) or None)

    return tool_calls, None


class StepRecord(BaseModel):
    """Per-step model-call record (#1483 contract); field names align with OpenAI shapes."""

    run_id: Optional[str] = None
    # step_index is always server-derived and authoritative. trial_index is set from a run's rollout
    # index (header, for Gym-orchestrated callers); turn_index is reserved for a future agent-loop
    # boundary and is currently always None. Both are None for base_url-only clients that correlate
    # via the URL prefix.
    trial_index: Optional[int] = None
    turn_index: Optional[int] = None
    step_index: int
    model_server: Optional[str] = None
    dialect: Optional[str] = None
    status_code: Optional[int] = None

    # Token accounting (aggregable to per-turn / per-trial).
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

    # Retry / error classification.
    error_category: Optional[str] = None
    retry_count: Optional[int] = None

    # Latency.
    latency_total_ms: Optional[float] = None
    latency_ttft_ms: Optional[float] = None


def build_step_record(exchange: dict[str, Any], *, step_index: int, run_id: Optional[str] = None) -> StepRecord:
    """Map one captured exchange (dialect/request/response + metadata) into a typed StepRecord."""
    response = exchange.get("response") or {}
    tokens = extract_token_stats(response.get("usage"))
    cache_hit, cached_tokens = _cache_signal(response.get("usage"))
    tool_calls, reasoning_content = _tool_calls_and_reasoning(response)
    return StepRecord(
        run_id=run_id or exchange.get("run_id"),
        trial_index=exchange.get("trial_index"),
        turn_index=exchange.get("turn_index"),
        step_index=step_index,
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
        retry_count=exchange.get("retry_count"),
        latency_total_ms=exchange.get("latency_ms"),
        **tokens,
    )


def assemble_step_records(store: CaptureStore, rollout_id: str, run_id: Optional[str] = None) -> list[StepRecord]:
    """Read a rollout's captured exchanges into ordered StepRecords (the per-trial trajectory)."""
    return [build_step_record(ex, step_index=i, run_id=run_id) for i, ex in enumerate(store.read(rollout_id))]


def aggregate_step_records(steps: list[StepRecord]) -> dict[str, Any]:
    """Per-rollout token / latency / turn totals from already-assembled StepRecords (#1483).

    Pure (no IO) so callers that already hold the steps don't re-read the capture file.
    """

    def _sum(attr: str) -> Optional[float]:
        values = [getattr(s, attr) for s in steps if getattr(s, attr) is not None]
        return sum(values) if values else None

    turns = {s.turn_index for s in steps if s.turn_index is not None}
    return {
        "tokens_in": _sum("tokens_in"),
        "tokens_out": _sum("tokens_out"),
        "tokens_reasoning": _sum("tokens_reasoning"),
        "tokens_total": _sum("tokens_total"),
        "latency_total_ms": _sum("latency_total_ms"),
        "num_turns": len(turns) if turns else (len(steps) or None),
        "num_steps": len(steps),
    }


def aggregate_rollout_metrics(store: CaptureStore, rollout_id: str) -> dict[str, Any]:
    """Per-rollout token / latency / turn totals, for the existing per-rollout record (#1483)."""
    return aggregate_step_records(assemble_step_records(store, rollout_id))
