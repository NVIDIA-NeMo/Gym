# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rows the bundle refuses: runtime features no supported row uses, and renamed tools it cannot hide.

A row that needs one is refused when the bundle is compiled, so it can never run with a feature silently missing:
``apply_patch``, and failed tool parts with ``nemo_gym_profile`` metadata (frozen error receipts) in a continuation
prefix. A row that renames tools (``tool_names`` aliases) is refused where OpenCode's own text would show the model the
built-in names: a selected tool without a description (OpenCode's descriptions name other built-in tools), and a
prefix's rejected calls kept in OpenCode's form (``unreceipted_rejected_calls``; OpenCode's error lists the built-in
tools). Several user messages and a ``max_output_tokens`` request field are refused by the OpenCode agent;
``instructions``, and a row that renames tools without a system message, by row validation (``task_variants.runtime``).
"""

from resources_servers.agentic_if.task_variants.interfaces import installed_interfaces
from resources_servers.agentic_if.task_variants.runtime import ContinuationSpec, RuntimeSpec
from resources_servers.agentic_if.task_variants.toolsets import renamed_tools


class UnsupportedFeatureError(ValueError):
    """The row needs a runtime feature the OpenCode plugin bundle does not implement."""


def _has_frozen_error_receipts(continuation: ContinuationSpec | None) -> bool:
    """Whether a continuation prefix has failed tool parts stamped with ``nemo_gym_profile`` metadata."""
    if continuation is None:
        return False
    return any(
        part.get("type") == "tool"
        and part.get("state", {}).get("status") == "error"
        and (part["state"].get("metadata") or {}).get("nemo_gym_profile")
        for message in continuation.prefix_export["messages"]
        for part in message.get("parts", [])
    )


def _check_renamed_tools(spec: RuntimeSpec) -> None:
    """Refuse a row that renames tools where OpenCode's own text would name the built-in tools."""
    selected = spec.tools.tool_names or {}
    described = set(spec.tools.tool_descriptions) | {
        tool
        for tool, definition in installed_interfaces(spec.tools.tool_interfaces).items()
        if "description" in definition
    }
    undescribed = sorted(set(selected) - described)
    if undescribed:
        raise UnsupportedFeatureError(
            "a row that renames tools must describe every tool it selects (OpenCode's own descriptions name the "
            f"built-in tools): {undescribed}"
        )
    if spec.continuation is not None and spec.continuation.unreceipted_rejected_calls:
        raise UnsupportedFeatureError(
            "a row that renames tools cannot keep a prefix's rejected calls in OpenCode's form "
            "(unreceipted_rejected_calls): OpenCode's error lists the built-in tools"
        )


def check_supported(spec: RuntimeSpec) -> None:
    """Raise ``UnsupportedFeatureError`` when ``spec`` needs a feature the bundle dropped."""
    if "apply_patch" in (spec.tools.tool_names or {}):
        raise UnsupportedFeatureError("selecting apply_patch is not supported by the OpenCode bundle")
    if _has_frozen_error_receipts(spec.continuation):
        raise UnsupportedFeatureError(
            "frozen error receipts in a continuation prefix are not supported by the OpenCode bundle"
        )
    if renamed_tools(spec.tools.tool_names):
        _check_renamed_tools(spec)
