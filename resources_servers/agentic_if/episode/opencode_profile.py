# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tool-profile records of the ``if-profile.mjs`` OpenCode plugin.

The plugin applies each tool's configured argument schema inside OpenCode and records the model's configured
arguments with every call; the boundary plugin applies the descriptions. These helpers read back what the model wrote
and saw. The profile's hash is ``opencode_bundle.profile.profile_sha256``.
"""

import re
from typing import Any, Mapping


PROFILE_METADATA_KEY = "nemo_gym_profile"
PROFILE_STASH_KEY = "__nemo_gym_profile"


def model_visible_tool_input(state: Mapping[str, Any]) -> Any:
    """The arguments of an exported OpenCode tool part as the model wrote them.

    OpenCode stores a tool part's input after the plugin mapped it to the built-in tool's form; the plugin records the
    model's configured arguments (extra arguments included) in the part's metadata. Without that record (no profile,
    or a tool that failed before the plugin's after-hook), the stored input is what the model sent.
    """
    metadata = state.get("metadata") if isinstance(state.get("metadata"), Mapping) else {}
    record = metadata.get(PROFILE_METADATA_KEY)
    if isinstance(record, Mapping) and "configured_args" in record:
        return record["configured_args"]
    stash = _stash(state)
    if "configured_args" in stash:
        return stash["configured_args"]
    return state.get("input") or {}


def model_visible_tool_output(state: Mapping[str, Any]) -> Any:
    """The result of an exported OpenCode tool part as the model saw it.

    That is the output or, for a failed call, the error text, with the built-in tool's argument names renamed to the
    profile's as the plugin showed them (if-profile.mjs toProfileNames).
    """
    stamp = (state.get("metadata") or {}).get(PROFILE_METADATA_KEY) or {}
    if state.get("status") == "error" and isinstance(stamp.get("model_error"), str):
        return stamp["model_error"]
    if state.get("output") is not None:
        return state["output"]
    error = state.get("error")
    names = _stash(state).get("names")
    if isinstance(error, str) and isinstance(names, Mapping):
        for builtin, configured in names.items():
            if builtin != configured:
                error = re.sub(rf"\b{re.escape(builtin)}\b", str(configured), error)
    return error


def _stash(state: Mapping[str, Any]) -> Mapping[str, Any]:
    """The plugin's record inside a reshaped call's stored input (if-profile.mjs STASH_KEY).

    It survives a call that fails inside the tool.
    """
    tool_input = state.get("input") if isinstance(state.get("input"), Mapping) else {}
    stash = tool_input.get(PROFILE_STASH_KEY)
    return stash if isinstance(stash, Mapping) else {}
