# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The profile plugin's receipts read back what the model wrote and saw; the profile hash names its configuration."""

import copy
import json

from resources_servers.agentic_if.episode.opencode_profile import model_visible_tool_input, model_visible_tool_output
from resources_servers.agentic_if.opencode_bundle.profile import profile_sha256


PROFILE = {
    "profile_version": 1,
    "id": "agentic-if",
    "tools": {"read": {"tool": "read", "args_map": {"filePath": "path"}, "parameters": {"type": "object"}}},
    "literal_errors": [],
}


def test_hash_ignores_key_order_and_existing_hash() -> None:
    reordered = json.loads(json.dumps(PROFILE, sort_keys=True))
    assert profile_sha256(reordered) == profile_sha256(PROFILE)
    assert profile_sha256({**PROFILE, "profile_sha256": "x"}) == profile_sha256(PROFILE)
    changed = copy.deepcopy(PROFILE)
    changed["tools"]["read"]["args_map"]["filePath"] = "file"
    assert profile_sha256(changed) != profile_sha256(PROFILE)


def test_model_view_prefers_the_plugin_receipt_then_its_stash_then_the_stored_input() -> None:
    stash = {"__nemo_gym_profile": {"configured_args": {"path": "a"}, "names": {"filePath": "path"}}}
    receipted = {"input": {"filePath": "a"}, "metadata": {"nemo_gym_profile": {"configured_args": {"path": "r"}}}}
    assert model_visible_tool_input(receipted) == {"path": "r"}
    assert model_visible_tool_input({"input": {"filePath": "a"} | stash}) == {"path": "a"}
    assert model_visible_tool_input({"input": {"filePath": "a"}}) == {"filePath": "a"}
    failed = {"status": "error", "input": {"filePath": "a"} | stash, "error": "filePath is required"}
    assert model_visible_tool_output(failed) == "path is required"
    presented = failed | {"metadata": {"nemo_gym_profile": {"model_error": "read failed: path is required"}}}
    assert model_visible_tool_output(presented) == "read failed: path is required"
