# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared model-request adaptation for host and sandbox Hermes runtimes (stdlib only)."""

import json
from types import SimpleNamespace
from typing import Any


def _model_api_kwargs(kwargs: dict[str, Any], *, preserve_reasoning_history: bool) -> dict[str, Any]:
    kwargs = kwargs.copy()
    extra_body = dict(kwargs.get("extra_body") or {})
    metadata = dict(kwargs.get("metadata") or {})
    template_kwargs = json.loads(metadata.pop("chat_template_kwargs", None) or "{}")
    template_kwargs.update(extra_body.pop("chat_template_kwargs", None) or {})
    # Thinking mode belongs to the Model Server; only history preservation is harness-owned.
    template_kwargs.pop("enable_thinking", None)
    if preserve_reasoning_history:
        template_kwargs["truncate_history_thinking"] = False
    # Gym rejects chat_template_kwargs at the top level, where the SDK expands extra_body.
    if template_kwargs:
        metadata["chat_template_kwargs"] = json.dumps(template_kwargs)
    if extra_body:
        kwargs["extra_body"] = extra_body
    else:
        kwargs.pop("extra_body", None)
    if metadata:
        kwargs["metadata"] = metadata
    else:
        kwargs.pop("metadata", None)
    return kwargs


def install_summary_compat(agent: Any, *, preserve_reasoning_history: bool) -> None:
    """Route pinned Hermes's iteration-limit summary through Gym's observed model path."""
    original_ensure_client = getattr(agent, "_ensure_primary_openai_client", None)
    original_handle_max_iterations = getattr(agent, "_handle_max_iterations", None)
    if not callable(original_ensure_client) or not callable(original_handle_max_iterations):
        return

    def create_summary(**kwargs: Any) -> Any:
        # Hermes's summary bypasses _build_api_kwargs and replays its internal history.
        # Copy before stripping fields so the recorded trajectory retains its full evidence.
        api_messages = []
        for message in kwargs["messages"]:
            api_message = {
                key: value
                for key, value in message.items()
                if key not in {"codex_reasoning_items", "reasoning", "finish_reason"}
            }
            api_messages.append(agent._sanitize_tool_calls_for_strict_api(api_message))
        request = _model_api_kwargs(
            {**kwargs, "messages": api_messages},
            preserve_reasoning_history=preserve_reasoning_history,
        )
        return agent._interruptible_api_call(request)

    summary_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_summary)))

    def ensure_client(*, reason: str) -> Any:
        if reason in {"iteration_limit_summary", "iteration_limit_summary_retry"}:
            return summary_client
        return original_ensure_client(reason=reason)

    def handle_max_iterations(messages: list[dict[str, Any]], api_call_count: int) -> str:
        agent._gym_iteration_limit_reached = True
        return original_handle_max_iterations(messages, api_call_count)

    agent._ensure_primary_openai_client = ensure_client
    agent._handle_max_iterations = handle_max_iterations
