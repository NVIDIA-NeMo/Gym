# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared model-request adaptation for host and sandbox Hermes runtimes (stdlib only)."""

import json
from typing import Any


class _SummaryCompatChildren(list):
    """Install Gym's summary adapter before a delegated child starts running."""

    _gym_child_hook_supported = True

    def append(self, child: Any) -> None:
        self._gym_prepare_child(child)
        super().append(child)


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


def configure_model_calls(agent: Any, *, preserve_reasoning_history: bool) -> None:
    """Adapt primary and delegated Hermes calls to Gym's model and training contracts."""
    if getattr(agent, "_gym_model_calls_configured", False):
        return
    agent._disable_streaming = agent._disable_streaming_seed = True
    original_build_kwargs = agent._build_api_kwargs
    original_build_message = agent._build_assistant_message
    original_handle_max_iterations = agent._handle_max_iterations

    def build_kwargs(api_messages: list, tools_for_api: list | None = None) -> dict[str, Any]:
        kwargs = _model_api_kwargs(
            original_build_kwargs(api_messages, tools_for_api=tools_for_api),
            preserve_reasoning_history=preserve_reasoning_history,
        )
        kwargs["stream"] = False
        return kwargs

    transport = agent._get_transport("chat_completions")
    original_normalize = transport.normalize_response

    summary_responses: list[dict[str, Any]] | None = None

    def normalize(response: Any, **kwargs: Any) -> Any:
        normalized = original_normalize(response, **kwargs)
        raw = response.choices[0].message
        fields = ("prompt_token_ids", "generation_token_ids", "generation_log_probs", "routed_experts")
        bundle = {key: value for key in fields if (value := getattr(raw, key, None)) is not None}
        if bundle:
            # NormalizedResponse keeps only recognized provider fields. Carry the data
            # on this exact response, so retries, summaries and child turns cannot swap it.
            normalized.provider_data = {**(normalized.provider_data or {}), "gym_training": bundle}
        if summary_responses is not None:
            summary_responses.append(bundle)
        return normalized

    transport.normalize_response = normalize

    def build_message(assistant_message: Any, finish_reason: str) -> dict[str, Any]:
        message = original_build_message(assistant_message, finish_reason)
        message.update((assistant_message.provider_data or {}).get("gym_training", {}))
        return message

    def handle_max_iterations(messages: list[dict[str, Any]], api_call_count: int) -> str:
        if not getattr(agent, "_gym_iteration_limit_reached", False):
            agent._gym_iteration_limit_reached = True
            callback = getattr(agent, "_gym_on_iteration_limit_reached", None)
            invocation_id = getattr(agent, "_gym_invocation_id", None)
            if callable(callback) and isinstance(invocation_id, str):
                callback(invocation_id=invocation_id)
        # The summary appends plain text directly instead of calling _build_assistant_message.
        # Only the successful final attempt owns the newly appended assistant turn.
        nonlocal summary_responses
        summary_responses = []
        original_length = len(messages)
        try:
            summary = original_handle_max_iterations(messages, api_call_count)
            if summary_responses and len(messages) > original_length and messages[-1].get("role") == "assistant":
                messages[-1].update(summary_responses[-1])
            return summary
        finally:
            summary_responses = None

    agent._build_api_kwargs = build_kwargs
    agent._build_assistant_message = build_message
    agent._handle_max_iterations = handle_max_iterations
    agent._gym_model_calls_configured = True

    children = getattr(agent, "_active_children", None)
    if isinstance(children, list):
        if not getattr(children, "_gym_child_hook_supported", False):
            children = _SummaryCompatChildren(children)
            agent._active_children = children
        children._gym_prepare_child = lambda child: configure_model_calls(
            child, preserve_reasoning_history=preserve_reasoning_history
        )
