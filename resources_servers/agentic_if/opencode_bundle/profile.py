# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``profile.json`` and ``plants.json``: the profile plugin's configuration, compiled from the RuntimeSpec.

The compiled profile is evidence format: its canonical hash ``profile_sha256`` is in the seed record, in the profile
plugin's load receipt and in every tool receipt, and the seed record keeps the digest of its staged bytes.
``tests/opencode_bundle/fixtures/profile-goldens.json`` pins them: keep every key and its order, including the keys
the plugin does not read (``profile_version``, ``id``, ``check_aliases``, ``prefix_receipts``, ``prefix_mode``).
"""

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pydantic import JsonValue

from resources_servers.agentic_if.opencode_bundle.features import check_supported
from resources_servers.agentic_if.task_variants.interfaces import installed_interfaces
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec


def profile_sha256(profile: Mapping[str, Any]) -> str:
    """SHA-256 of the profile in canonical JSON (sorted keys, no whitespace), without its own hash."""
    body = {key: value for key, value in profile.items() if key != "profile_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compile_profile(spec: RuntimeSpec) -> dict[str, JsonValue]:
    """The profile plugin's configuration: installed interfaces, read-first guards and literal error texts.

    Execution sees the built-in tool IDs; presentation (descriptions, parameter descriptions) belongs to the boundary,
    so interface descriptions are removed here. ``visible_tools`` keeps the selection for error presentation.
    """
    check_supported(spec)
    definitions = installed_interfaces(spec.tools.tool_interfaces)
    for definition in definitions.values():
        definition.pop("description", None)
    config: dict[str, JsonValue] = {"profile_version": 1, "id": "agentic-if", "tools": definitions}
    if spec.tools.read_first is not None:
        config["read_first"] = spec.tools.read_first.model_dump(mode="json")
    if spec.tools.tool_names is not None:
        config["visible_tools"] = dict(spec.tools.tool_names)
        # Unread hash input: check_supported refuses aliases.
        config["check_aliases"] = False
    config["literal_errors"] = [event.text for event in spec.events]
    if spec.continuation is not None:
        # Unread hash inputs; prefix_receipts is always empty (check_supported refuses frozen error receipts).
        config["prefix_receipts"] = []
        config["prefix_mode"] = spec.continuation.mode
    config["profile_sha256"] = profile_sha256(config)
    return config


def compile_plants(spec: RuntimeSpec) -> list[dict[str, JsonValue]]:
    """The planted tool failures, in the form the profile plugin reads."""
    return [event.model_dump(mode="json", exclude_none=True) for event in spec.events]
