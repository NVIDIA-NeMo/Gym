# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rendering, parsing, RNG, and permissive tool validation."""

from __future__ import annotations

import json
import random
import re
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import jsonschema.validators
from json_repair import loads as json_repair_loads
from pydantic import BaseModel, ConfigDict

from responses_api_agents.conversational_tool_use_policy_tool_generation.assets import GoldenPair


class Tau2CompatibleToolSignature(BaseModel):
    """Permissive Tau2-compatible tool signature."""

    model_config = ConfigDict(extra="allow")

    name: str
    doc: str
    params: dict[str, Any] | None = None
    returns: dict[str, Any] | None = None


def format_domain_name(name: str) -> str:
    return name.replace("(", "").replace(")", "").replace("/", "-").replace(" ", "_").replace("&", "")


def sample_timestamp() -> str:
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
    random_naive = start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))
    timezone_name = random.choices(
        [name for name, _ in timezones],
        weights=[weight for _, weight in timezones],
        k=1,
    )[0]
    return random_naive.replace(tzinfo=ZoneInfo(timezone_name)).strftime("%Y-%m-%d %H:%M:%S %Z")


def shuffled_pairs(pairs: tuple[GoldenPair, ...]) -> list[GoldenPair]:
    shuffled = list(pairs)
    random.shuffle(shuffled)
    return shuffled


def format_policy_tool_pair(policy: str, tools: str, index: int) -> str:
    return f"\n\n<policy_{index}>\n{policy}\n</policy_{index}>\n<tools_{index}>\n{tools}\n</tools_{index}>"


def format_policy_reference(policy: str, index: int) -> str:
    return f"\n\n<policy_{index}>\n{policy}\n</policy_{index}>"


def policy_tool_references(pairs: list[GoldenPair]) -> str:
    return "".join(format_policy_tool_pair(pair.policy, pair.tools, index) for index, pair in enumerate(pairs))


def policy_references(pairs: list[GoldenPair]) -> str:
    return "".join(format_policy_reference(pair.policy, index) for index, pair in enumerate(pairs))


def parse_tag(text: str, tag: str) -> str | None:
    try:
        matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        if not matches:
            return None
        return matches[-1].strip()
    except Exception:
        return None


def parse_policy(text: str) -> str | None:
    return parse_tag(text, "policy")


def parse_tools(text: str) -> list[Any] | None:
    try:
        tagged = parse_tag(text, "tools")
        return [json_repair_loads(line) for line in tagged.strip().split("\n") if line.strip()]
    except Exception:
        return None


def parse_judgment(text: str) -> Any:
    try:
        tagged = parse_tag(text, "judgment")
        return json_repair_loads(tagged)
    except Exception:
        return False


def serialize_tools(tools: list[Any]) -> str:
    return "\n".join(json.dumps(tool) for tool in tools)


def tools_artifact(tools: list[Any]) -> str:
    return "".join(json.dumps(tool) + "\n" for tool in tools)


def _validate_json_schema(schema: dict[str, Any] | None) -> None:
    if schema is None:
        return
    validator = jsonschema.validators.validator_for(schema)
    validator.check_schema(schema)


def validate_tools(tools: list[Any]) -> bool:
    try:
        signatures = [Tau2CompatibleToolSignature.model_validate(tool) for tool in tools]
        for signature in signatures:
            _validate_json_schema(signature.params)
            _validate_json_schema(signature.returns)
        return True
    except Exception:
        return False
