# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime features the bundle refuses because no supported row uses them.

A row that needs one of these is refused when the bundle is compiled, so it can never run with a feature silently
missing. Aliases, ``apply_patch`` and frozen error receipts in a continuation prefix are checked here; several user
messages, ``max_output_tokens`` and ``instructions`` are request fields the OpenCode agent already rejects.
"""

from copy import deepcopy

from pydantic import JsonValue

from resources_servers.agentic_if.task_variants.runtime import ContinuationSpec, RuntimeSpec


class UnsupportedFeatureError(ValueError):
    """The row needs a runtime feature the OpenCode plugin bundle does not implement."""


def prefix_receipts(continuation: ContinuationSpec | None) -> list[dict[str, JsonValue]]:
    """Frozen error metadata in a continuation prefix (``nemo_gym_profile`` stamps on failed tool parts).

    Moved from ``drivers/opencode_hooks.py``. The bundle does not restamp them, so a non-empty list is refused;
    the empty list stays in the compiled profile so ``profile_sha256`` is unchanged for continuation rows.
    """
    if continuation is None:
        return []
    return [
        {"call_id": part["callID"], "opencode_tool": part["tool"], "metadata": deepcopy(stamp)}
        for message in continuation.prefix_export["messages"]
        for part in message.get("parts", [])
        if part.get("type") == "tool" and part.get("state", {}).get("status") == "error"
        if (stamp := (part["state"].get("metadata") or {}).get("nemo_gym_profile"))
    ]


def check_supported(spec: RuntimeSpec) -> None:
    """Raise ``UnsupportedFeatureError`` when ``spec`` needs a feature the bundle dropped."""
    names = spec.tools.tool_names or {}
    aliased = sorted(builtin for builtin, visible in names.items() if builtin != visible)
    if aliased:
        raise UnsupportedFeatureError(f"tool aliases are not supported by the OpenCode bundle: {aliased}")
    if "apply_patch" in names:
        raise UnsupportedFeatureError("selecting apply_patch is not supported by the OpenCode bundle")
    if prefix_receipts(spec.continuation):
        raise UnsupportedFeatureError(
            "frozen error receipts in a continuation prefix are not supported by the OpenCode bundle"
        )
