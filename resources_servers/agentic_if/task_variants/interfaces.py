# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Installed interfaces of OpenCode's built-in tools, not executable mappings supplied by task rows."""

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _catalogue() -> dict:
    return json.loads(Path(__file__).with_name("data").joinpath("opencode_interfaces.json").read_text())


def installed_interfaces(selection: dict[str, str]) -> dict:
    """Resolve selected tool -> interface ID to independent trusted definitions."""
    result = {}
    for tool, identity in selection.items():
        if not isinstance(identity, str) or identity not in _catalogue():
            raise ValueError(f"unknown installed interface for {tool}: {identity!r}")
        item = _catalogue()[identity]
        if item["tool"] != tool:
            raise ValueError(f"interface {identity} belongs to {item['tool']}, not {tool}")
        result[tool] = deepcopy(item) | {"id": identity}
    return result
