# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime features the bundle refuses because no supported row uses them.

A row that needs one is refused when the bundle is compiled, so it can never run with a feature silently missing:
tool aliases, ``apply_patch``, and failed tool parts with ``nemo_gym_profile`` metadata (frozen error receipts) in a
continuation prefix. Several user messages and a ``max_output_tokens`` request field are refused by the OpenCode agent,
``instructions`` by row validation (``task_variants.runtime``).
"""

from resources_servers.agentic_if.task_variants.runtime import ContinuationSpec, RuntimeSpec


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


def check_supported(spec: RuntimeSpec) -> None:
    """Raise ``UnsupportedFeatureError`` when ``spec`` needs a feature the bundle dropped."""
    names = spec.tools.tool_names or {}
    aliased = sorted(builtin for builtin, visible in names.items() if builtin != visible)
    if aliased:
        raise UnsupportedFeatureError(f"tool aliases are not supported by the OpenCode bundle: {aliased}")
    if "apply_patch" in names:
        raise UnsupportedFeatureError("selecting apply_patch is not supported by the OpenCode bundle")
    if _has_frozen_error_receipts(spec.continuation):
        raise UnsupportedFeatureError(
            "frozen error receipts in a continuation prefix are not supported by the OpenCode bundle"
        )
