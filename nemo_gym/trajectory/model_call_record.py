# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One model call as a rollout records it: token counts, cache use, tool calls, and totals across calls."""

import json
from typing import Any, Literal, Mapping, Optional

from pydantic import BaseModel, Field

from nemo_gym.config.config_types import ModelServerRef


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

    ``tokens_in`` is the whole prompt size (see ``_prompt_size``); cache creation is also reported
    separately as ``cache_creation_tokens``. A count the provider did not report stays ``None``, never 0.

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
    tokens_in = _prompt_size(usage)
    tokens_out = _token_count(usage.get("output_tokens"))
    if tokens_out is None:
        tokens_out = _token_count(usage.get("completion_tokens"))
    cache_creation = _token_count(usage.get("cache_creation_input_tokens"))
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
            elif item.get("type") == "custom_tool_call":
                # A free-form tool takes one raw string; record it the way a function stand-in takes it.
                tool_calls.append(
                    {
                        "call_id": item.get("call_id") or item.get("id"),
                        "name": item.get("name"),
                        "arguments": {"input": item.get("input")},
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
    client_session_id: Optional[str] = Field(
        default=None,
        description="Client-declared correlation identifier; evidence only, not a security boundary.",
    )
    client_assistant_message_id: Optional[str] = Field(
        default=None,
        description="Persisted client assistant-message ID; retries may share it, model_call_id remains per attempt.",
    )

    # Durable append order, not a causal or semantic order for concurrent calls.
    call_index: int
    model_ref: Optional[ModelServerRef] = None
    model: Optional[str] = None
    dialect: Optional[str] = None
    status_code: Optional[int] = None
    response_status: Optional[str] = None
    finish_reason: Optional[str] = None
    upstream_attempted: Optional[bool] = None
    response_source: Optional[Literal["upstream", "local"]] = None
    upstream_status_code: Optional[int] = None
    local_response_reason: Optional[str] = None

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
        client_session_id=(
            exchange.get("client_session_id") if isinstance(exchange.get("client_session_id"), str) else None
        ),
        client_assistant_message_id=(
            exchange.get("client_assistant_message_id")
            if isinstance(exchange.get("client_assistant_message_id"), str)
            else None
        ),
        call_index=call_index,
        model_ref=exchange.get("model_ref"),
        model=model if isinstance(model, str) else None,
        dialect=exchange.get("dialect"),
        status_code=exchange.get("status_code"),
        response_status=response.get("status") if isinstance(response.get("status"), str) else None,
        finish_reason=finish_reason,
        upstream_attempted=exchange.get("upstream_attempted"),
        response_source=exchange.get("response_source"),
        upstream_status_code=exchange.get("upstream_status_code"),
        local_response_reason=exchange.get("local_response_reason"),
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


def _prompt_size(usage: Mapping[str, Any]) -> Optional[int]:
    """Return the whole prompt size, with cached tokens counted exactly once.

    Usage formats disagree on whether the prompt count includes the cache:

    - ``prompt_tokens`` (Chat Completions) includes it. It wins whenever present, also when a gateway
      adds Anthropic-style ``cache_*`` fields beside it.
    - ``input_tokens`` with top-level ``cache_read_input_tokens`` or ``cache_creation_input_tokens``
      (Anthropic Messages) excludes it, so the cache counts are added.
    - ``input_tokens`` alone (Responses) includes it.
    """
    prompt_tokens = _token_count(usage.get("prompt_tokens"))
    if prompt_tokens is not None:
        return prompt_tokens
    input_tokens = _token_count(usage.get("input_tokens"))
    cache_read = _token_count(usage.get("cache_read_input_tokens"))
    cache_creation = _token_count(usage.get("cache_creation_input_tokens"))
    if cache_read is None and cache_creation is None:
        return input_tokens
    cache_total = (cache_read or 0) + (cache_creation or 0)
    # A fully cached Anthropic response may omit input_tokens; zero cache counts alone prove no prompt size.
    if input_tokens is None and cache_total == 0:
        return None
    return (input_tokens or 0) + cache_total
