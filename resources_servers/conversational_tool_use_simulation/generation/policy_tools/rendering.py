# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt input formatting for policy and tool generation."""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo


def format_domain_name(name: str) -> str:
    return name.replace("(", "").replace(")", "").replace("/", "-").replace(" ", "_").replace("&", "")


def sample_timestamp(rng: random.Random) -> str:
    timezones = [
        ("America/New_York", 0.47),
        ("America/Chicago", 0.33),
        ("America/Denver", 0.06),
        ("America/Los_Angeles", 0.13),
        ("America/Anchorage", 0.003),
        ("Pacific/Honolulu", 0.004),
        ("America/Phoenix", 0.01),
    ]
    start = datetime(2025, 1, 1, 0, 0, 0)
    end = datetime(2025, 12, 31, 23, 59, 59)
    random_naive = start + timedelta(seconds=rng.randint(0, int((end - start).total_seconds())))
    timezone = ZoneInfo(
        rng.choices([name for name, _ in timezones], weights=[weight for _, weight in timezones], k=1)[0]
    )
    return random_naive.replace(tzinfo=timezone).strftime("%Y-%m-%d %H:%M:%S %Z")


def serialize_tools(tools: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(tool) for tool in tools)


def format_policy_tool_pair(policy: str, tools: str, index: int) -> str:
    return f"\n\n<policy_{index}>\n{policy}\n</policy_{index}>\n<tools_{index}>\n{tools}\n</tools_{index}>"


def shuffled_policy_tool_references(pairs: list[tuple[str, str]], rng: random.Random) -> str:
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    return "".join(format_policy_tool_pair(policy, tools, index) for index, (policy, tools) in enumerate(shuffled))


def shuffled_policy_references(pairs: list[tuple[str, str]], rng: random.Random) -> str:
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    return "".join(f"\n\n<policy_{index}>\n{policy}\n</policy_{index}>" for index, (policy, _) in enumerate(shuffled))


def advance_reference_shuffle(pairs: list[tuple[str, str]], rng: random.Random) -> None:
    """Advance the deterministic shuffle stream between refinement phases."""
    shuffled = list(pairs)
    rng.shuffle(shuffled)
