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


def _keep_sampled_continuation_fragments(collapse: Any) -> Any:
    """Keep each sampled continuation fragment when Hermes stops continuing a truncated reply.

    At its continuation ceiling Hermes replaces the turn's fragment rows, which carry Gym's token IDs and log
    probabilities, with one stitched assistant row that has none. The turn ends right after, so restore the rows
    as a recovered continuation leaves them: fragments and continuation prompts, unflagged.
    """

    def collapse_and_keep(
        agent: Any, messages: list[Any], current_turn_user_idx: Any, *, finish_reason: str, parts: Any = None
    ) -> str:
        valid_index = isinstance(current_turn_user_idx, int) and current_turn_user_idx >= 0
        start = current_turn_user_idx + 1 if valid_index else 0
        before = list(messages[start:])
        partial = collapse(agent, messages, current_turn_user_idx, finish_reason=finish_reason, parts=parts)
        # Only the ceiling ("length") ends the turn here; the error path's trail may end on an unanswered prompt.
        if finish_reason != "length" or not any(
            isinstance(row, dict) and row.get("_length_continuation_fragment") for row in before
        ):
            return partial
        for row in before:
            if isinstance(row, dict):
                row.pop("_length_continuation_fragment", None)
                row.pop("_length_continuation_nudge", None)
        messages[start:] = before
        agent._session_messages = messages
        return partial

    collapse_and_keep._gym_keeps_fragments = True
    return collapse_and_keep


def _install_continuation_fragment_capture() -> None:
    try:
        from agent import turn_truncation
    except ImportError:  # Not running inside the pinned Hermes runtime.
        return
    if not getattr(turn_truncation.collapse_continuation_trail, "_gym_keeps_fragments", False):
        turn_truncation.collapse_continuation_trail = _keep_sampled_continuation_fragments(
            turn_truncation.collapse_continuation_trail
        )


def configure_model_calls(agent: Any, *, preserve_reasoning_history: bool) -> None:
    """Adapt primary and delegated Hermes calls to Gym's model and training contracts."""
    if getattr(agent, "_gym_model_calls_configured", False):
        return
    agent._disable_streaming = agent._disable_streaming_seed = True
    _install_continuation_fragment_capture()
    original_build_kwargs = agent._build_api_kwargs
    original_build_message = agent._build_assistant_message
    original_handle_max_iterations = agent._handle_max_iterations

    def build_kwargs(api_messages: list, tools_for_api: list | None = None) -> dict[str, Any]:
        kwargs = _model_api_kwargs(
            original_build_kwargs(api_messages, tools_for_api=tools_for_api),
            preserve_reasoning_history=preserve_reasoning_history,
        )
        kwargs["stream"] = False
        # Hermes doubles the output cap on each truncation retry, up to 32,768 tokens. Gym's `max_tokens` is the
        # per-call budget that runs are compared and trained under, so keep every call within it.
        configured = getattr(agent, "max_tokens", None)
        if isinstance(configured, int):
            for key in ("max_tokens", "max_completion_tokens"):
                if isinstance(kwargs.get(key), int) and kwargs[key] > configured:
                    kwargs[key] = configured
        return kwargs

    summary_responses: list[dict[str, Any]] | None = None

    def capture_token_data(original_normalize: Any) -> Any:
        def normalize(response: Any, **kwargs: Any) -> Any:
            return attach_token_data(response, original_normalize(response, **kwargs))

        return normalize

    def attach_token_data(response: Any, normalized: Any) -> Any:
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

    original_get_transport = agent._get_transport

    def get_transport(api_mode: str | None = None) -> Any:
        # Hermes clears its transport cache when it rebuilds the client after transient errors, so wrap every
        # transport it creates, not only the first cached one, or later turns lose their token IDs.
        transport = original_get_transport(api_mode)
        if (
            (api_mode or agent.api_mode) == "chat_completions"
            and transport is not None
            and not getattr(transport, "_gym_token_capture", False)
        ):
            transport.normalize_response = capture_token_data(transport.normalize_response)
            transport._gym_token_capture = True
        return transport

    agent._get_transport = get_transport
    get_transport("chat_completions")

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
