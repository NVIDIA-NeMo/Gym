# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-content facts for framework-owned context decisions.

This module records what the gateway received and exposed. It does not decide
whether to splice tokens, compact a context, or create a training segment.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from nemo_gym.token_id_capture.records import strip_token_fields
from nemo_gym.token_id_capture.staging.records import ReplayContext, ReplayItem, ReplaySummary


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(b"nemo-gym-replay-v1\0" + encoded.encode()).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _normalize_arguments(value: dict) -> None:
    if isinstance(value.get("arguments"), str):
        try:
            value["arguments"] = json.loads(value["arguments"])
        except json.JSONDecodeError:
            pass  # Invalid JSON remains exact prompt content.


def replay_context(items: list[dict], *, render_digest: str | None = None) -> ReplayContext:
    """Hash ordered source items without discarding exposed reasoning content.

    Only documented non-prompt response metadata is ignored. Unknown item kinds
    fail instead of collapsing to a fingerprint that could restore edited text.
    Media URLs/data are hashed here, never copied into the capture manifest.
    """
    cleaned, _ = strip_token_fields(items)
    result = []
    for item in cleaned:
        value = _plain(item)
        kind = value.get("type", "message")
        role = value.get("role")
        if kind == "message":
            role = "envelope" if role == "_ng_request_envelope" else role
            if role not in ("system", "developer", "user", "assistant", "tool", "envelope"):
                raise ValueError("Unsupported replay message role")
            value.pop("type", None)
        elif kind in ("reasoning", "function_call"):
            role = "assistant"
        elif kind == "function_call_output":
            role = "tool"
        else:
            raise ValueError(f"Unsupported replay item kind: {kind}")
        value.pop("id", None)
        value.pop("status", None)
        if kind == "function_call":
            _normalize_arguments(value)
        elif kind == "message" and role == "assistant":
            for call in value.get("tool_calls") or []:
                if isinstance(call, dict) and isinstance(call.get("function"), dict):
                    _normalize_arguments(call["function"])
        digest = _digest(value)
        alternate = None
        if (
            kind == "message"
            and role == "assistant"
            and value.get("tool_calls")
            and value.get("content") in (None, "")
        ):
            # Keep strict identity intact. Only vLLM's HF/string path is known
            # to render these two Chat tool-message forms identically.
            alternate = _digest({**value, "content": ""})
        result.append(ReplayItem(role=role, digest=digest, empty_tool_content_digest=alternate))
    return ReplayContext(items=result, render_digest=render_digest)


def summarize_replay(context: ReplayContext, *, item_count: int | None = None) -> ReplaySummary:
    """Summarize a complete source history or a prefix of the current request.

    Roles and item order are authenticated along with the normalized content.
    Reject an unavailable prefix rather than silently hashing a shorter slice.
    """
    count = len(context.items) if item_count is None else item_count
    if not 0 <= count <= len(context.items):
        raise ValueError("Replay prefix length is outside the supplied source history")
    return ReplaySummary(
        item_count=count,
        source_digest=_digest(
            {
                "normalization_version": context.normalization_version,
                "items": [(item.role, item.digest) for item in context.items[:count]],
            }
        ),
        empty_tool_content_digest=_digest(
            {
                "normalization_version": context.normalization_version,
                "items": [
                    (item.role, item.empty_tool_content_digest or item.digest) for item in context.items[:count]
                ],
            }
        ),
        render_digest=context.render_digest,
    )


def same_replay_source(left: ReplaySummary, right: ReplaySummary, *, allow_empty_tool_content: bool = False) -> bool:
    """Compare source facts; render compatibility remains the worker's decision."""
    return (
        left.normalization_version == right.normalization_version
        and left.item_count == right.item_count
        and (
            left.source_digest == right.source_digest
            or (
                allow_empty_tool_content
                and left.empty_tool_content_digest is not None
                and left.empty_tool_content_digest == right.empty_tool_content_digest
            )
        )
    )


def render_options_digest(payload: dict[str, Any]) -> str:
    """Identify the effective gateway options that can shape a rendered prompt."""
    fields = (
        "model",
        "chat_template",
        "chat_template_kwargs",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "documents",
        "add_generation_prompt",
        "continue_final_message",
        "add_special_tokens",
        "mm_processor_kwargs",
        "reasoning_effort",
        "media_io_kwargs",
    )
    return _digest({key: _plain(payload[key]) for key in fields if key in payload})
