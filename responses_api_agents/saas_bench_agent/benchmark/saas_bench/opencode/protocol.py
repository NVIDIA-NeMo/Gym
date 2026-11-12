# SPDX-License-Identifier: Apache-2.0
# Source: https://gitlab-master.nvidia.com/pritamb/saas-bench/-/blob/31177a7a80040edf588388dd982af4b678d1e544/saas_bench/opencode/protocol.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Gym adaptation: preserve task checks while separating model and evaluation failures.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep the Gym/OpenCode boundary native and detect incomplete model replies."""

from __future__ import annotations

import re


class ModelProtocolError(ValueError):
    """The model intended an action but did not return an executable tool call."""


def native_message(
    message: dict, *, tools_available: bool, finish_reason: str | None = None
) -> dict:
    """Undo Gym's reasoning wrapper without executing tool-like reasoning text."""
    result = dict(message)
    if len(result.get("tool_calls") or []) > 32:
        raise ModelProtocolError(
            "Model returned more than 32 tool calls in one turn; retry with a smaller batch. "
            "No calls from this response were executed."
        )
    content = result.get("content") or ""
    if not isinstance(content, str):
        return result
    match = re.match(r"^<think>\n?(.*?)</think>(.*)$", content, re.DOTALL)
    if match:
        reasoning, content = match.groups()
        result["reasoning_content"] = reasoning
        result["reasoning"] = reasoning
        result["content"] = content or None
    if tools_available and not result.get("tool_calls"):
        if finish_reason == "length":
            # Preserve the provider's output limit for native OpenCode. An
            # exhausted generation is not malformed, and reasoning text is
            # still never interpreted as an executable action.
            return result
        if re.search(
            r"<tool_call>.*</tool_call>\s*(?:</think>)?\s*$",
            message.get("content") or "",
            re.DOTALL,
        ):
            raise ModelProtocolError(
                "Model returned tool-call markup as text instead of structured tool_calls; retry generation. "
                "Do not execute actions embedded in reasoning."
            )
        if not content.strip():
            raise ModelProtocolError(
                "Model returned only reasoning, with no answer or structured action; retry generation"
            )
    return result


def execution_health(events: list[dict], *, timed_out: bool = False) -> dict:
    """Report native execution independently from task score and process exit."""
    from saas_bench.opencode.core import context_overflow_recovered

    errors = [event for event in events if event.get("type") == "error"]
    recovered = context_overflow_recovered(events) if errors else False
    tools = [e.get("part", {}) for e in events if e.get("type") == "tool_use"]
    failed = [t for t in tools if t.get("state", {}).get("status") == "error"]
    final = next(
        (
            e.get("part", {}).get("text", "")
            for e in reversed(events)
            if e.get("type") == "text"
        ),
        "",
    )
    protocol_errors = any(
        "Model returned tool-call markup as text" in str(e)
        or "Model returned only reasoning" in str(e)
        or "Model returned more than 32 tool calls" in str(e)
        or "Model violated the native tool protocol" in str(e)
        for e in events
        if e.get("type") == "error"
    )
    malformed = bool(re.search(r"<tool_call>.*</tool_call>", final, re.DOTALL))
    last_finish = next(
        (e.get("part", {}) for e in reversed(events) if e.get("type") == "step_finish"),
        {},
    )
    output_limit = last_finish.get("reason") == "length"
    termination = (
        "timeout"
        if timed_out
        else "model_protocol_error"
        if protocol_errors
        else "context_limit"
        if not recovered and any(e.get("error", {}).get("name") == "ContextOverflowError" for e in errors)
        else "native_error"
        if errors and not recovered
        else "output_token_limit"
        if output_limit
        else "malformed_tool_call"
        if malformed
        else "final_response"
    )
    return {
        "tool_calls": len(tools),
        "tool_errors": len(failed),
        "termination": termination,
        "execution_healthy": not failed and termination == "final_response",
        "native_error_events": len(errors),
        "recovered_context_overflow": recovered,
        "output_token_limit": output_limit,
        "task_success": None,
    }
