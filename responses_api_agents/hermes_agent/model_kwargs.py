# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared model-request adaptation for host and sandbox Hermes runtimes (stdlib only)."""

import json
from types import SimpleNamespace
from typing import Any


class _SummaryCompatChildren(list):
    """Install Gym's summary adapter before a delegated child starts running."""

    _gym_child_hook_supported = True

    def append(self, child: Any) -> None:
        self._gym_prepare_child(child)
        super().append(child)


# Besides every 5xx, the 4xx a resend may clear: request timeout, too early, too large (Hermes compresses and sends a
# smaller request) and rate limit.
_RESENDABLE_CLIENT_ERROR_STATUSES = frozenset({408, 413, 425, 429})


class ModelRequestRejected(ValueError):
    """A model request rejected with a 4xx that sending it again cannot fix; keeps the SDK error's text and status.

    Pinned Hermes resends every OpenAI SDK 400 as transient: it reads the error message one level deeper than the
    SDK keeps it. It ends the run at once on a ValueError, after its context-length and 413 checks.
    """

    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.status_code = error.status_code
        self.body = getattr(error, "body", None)
        self.response = getattr(error, "response", None)


def install_model_retry_policy(agent: Any) -> None:
    """Resend a failed model request only for 408, 425, 429, 5xx and transport errors (pinned Hermes 26bb847a).

    Hermes's own loop resends 429, 5xx and transport errors three times, 2 s and 4 s apart. The OpenAI SDK under it
    would also resend each of them, and a 409, twice more.
    """
    if getattr(agent, "_gym_model_retry_policy_installed", False):
        return
    client_kwargs = getattr(agent, "_client_kwargs", None)
    api_call = getattr(agent, "_interruptible_api_call", None)
    if not isinstance(client_kwargs, dict) or not callable(api_call):
        return
    # Every request copies these into its own client.
    client_kwargs["max_retries"] = 0

    def call_model(*args: Any, **kwargs: Any) -> Any:
        try:
            return api_call(*args, **kwargs)
        except Exception as error:
            status = getattr(error, "status_code", None)
            if isinstance(status, int) and 400 <= status < 500 and status not in _RESENDABLE_CLIENT_ERROR_STATUSES:
                raise ModelRequestRejected(error) from error
            raise

    agent._interruptible_api_call = call_model
    agent._gym_model_retry_policy_installed = True


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
    if getattr(agent, "_gym_summary_compat_installed", False):
        return
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
        if not getattr(agent, "_gym_iteration_limit_reached", False):
            agent._gym_iteration_limit_reached = True
            callback = getattr(agent, "_gym_on_iteration_limit_reached", None)
            invocation_id = getattr(agent, "_gym_invocation_id", None)
            if callable(callback) and isinstance(invocation_id, str):
                callback(invocation_id=invocation_id)
        return original_handle_max_iterations(messages, api_call_count)

    agent._ensure_primary_openai_client = ensure_client
    agent._handle_max_iterations = handle_max_iterations
    agent._gym_summary_compat_installed = True

    children = getattr(agent, "_active_children", None)
    if isinstance(children, list):
        if not getattr(children, "_gym_child_hook_supported", False):
            children = _SummaryCompatChildren(children)
            agent._active_children = children

        def prepare_child(child: Any) -> None:
            install_summary_compat(child, preserve_reasoning_history=preserve_reasoning_history)
            install_model_retry_policy(child)

        children._gym_prepare_child = prepare_child
