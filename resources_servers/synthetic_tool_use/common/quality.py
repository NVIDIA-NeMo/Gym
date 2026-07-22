# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic validation for generated seed artifacts."""

from typing import Any

import jsonschema.validators
from jsonschema.exceptions import SchemaError

from resources_servers.synthetic_tool_use.common.models import (
    CustomerScenarioArtifact,
    SeedToolSignature,
)


SCHEMA_MARKERS = {"type", "properties", "required", "items", "enum", "anyOf", "oneOf", "allOf", "$ref", "title"}
DSML_LEAK_MARKERS = (
    "DSML",
    "\\uff5cDSML\\uff5c",
    "｜DSML｜",
    "<｜DSML｜function_calls",
    "</｜DSML｜function_calls>",
    "<｜DSML｜invoke",
    "</｜DSML｜invoke>",
    "<｜DSML｜parameter",
    "</｜DSML｜parameter>",
)
SPECIAL_TOKEN_LEAK_MARKERS = (
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<think>",
    "</think>",
    "<function_results>",
    "</function_results>",
    "<result>",
    "</result>",
)


class ArtifactValidationError(ValueError):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def detect_leak(text: str) -> str | None:
    if any(marker in text for marker in DSML_LEAK_MARKERS):
        return "dsml_leak"
    if any(marker in text for marker in SPECIAL_TOKEN_LEAK_MARKERS):
        return "special_token_leak"
    return None


def reject_leaks(text: str, *, artifact_name: str) -> None:
    reason = detect_leak(text)
    if reason:
        raise ArtifactValidationError(reason, f"{reason} in {artifact_name}")


def validate_tool_schema(value: Any, *, tool_name: str, field_name: str) -> None:
    reason = f"invalid_tool_{field_name}_schema"
    if not isinstance(value, dict):
        raise ArtifactValidationError(reason, f"tool {tool_name} has non-dict {field_name} schema")
    if not value or not SCHEMA_MARKERS.intersection(value):
        raise ArtifactValidationError(reason, f"tool {tool_name} has non-schema {field_name}")
    try:
        validator_class = jsonschema.validators.validator_for(value)
        validator_class.check_schema(value)
    except SchemaError as exc:
        raise ArtifactValidationError(
            reason,
            f"tool {tool_name} has invalid {field_name} JSON Schema at {exc.json_path}: {exc.message}",
        ) from exc


def validate_tools(raw_tools: list[dict[str, Any]]) -> list[SeedToolSignature]:
    if not raw_tools:
        raise ArtifactValidationError("empty_tools", "tools contain no entries")
    tools: list[SeedToolSignature] = []
    seen_names: set[str] = set()
    for index, raw_tool in enumerate(raw_tools):
        try:
            tool = SeedToolSignature.model_validate(raw_tool)
        except ValueError as exc:
            raise ArtifactValidationError("invalid_tool", f"tool {index}: {exc}") from exc
        if tool.name in seen_names:
            raise ArtifactValidationError("duplicate_tool_name", f"duplicate tool name: {tool.name}")
        seen_names.add(tool.name)
        validate_tool_schema(tool.params, tool_name=tool.name, field_name="params")
        validate_tool_schema(tool.returns, tool_name=tool.name, field_name="returns")
        reject_leaks(tool.model_dump_json(), artifact_name=f"tool {tool.name}")
        tools.append(tool)
    return tools


def validate_scenario(raw_scenario: dict[str, Any]) -> CustomerScenarioArtifact:
    try:
        scenario = CustomerScenarioArtifact.model_validate(raw_scenario)
    except ValueError as exc:
        raise ArtifactValidationError("invalid_scenario", str(exc)) from exc
    reject_leaks(scenario.model_dump_json(), artifact_name="scenario")
    return scenario
