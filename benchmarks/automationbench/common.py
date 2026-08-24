# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared types and state helpers for the AutomationBench V1 taskset."""

from __future__ import annotations

import verifiers.v1 as vf
from pydantic import Field

from automationbench.schema.world import WorldState


# A single tool server can expose AutomationBench's familiar bare API tool names.
TOOL_PREFIX = None


class AutomationBenchState(vf.State):
    """Per-rollout score mirrored from the live WorldState tool server."""

    partial_credit: float | None = None


class AutomationBenchData(vf.TaskData):
    domain: str
    assertions: list[dict]
    initial_state: dict
    zapier_tools: list[str] = Field(default_factory=list)
    source_id: int | str | None = None


class AutomationBenchToolsetConfig(vf.ToolsetConfig):
    pass


def strip_none_values(obj):
    """Remove HuggingFace-added ``None`` values before WorldState validation."""
    if isinstance(obj, dict):
        return {key: strip_none_values(value) for key, value in obj.items() if value is not None}
    if isinstance(obj, list):
        return [strip_none_values(value) for value in obj if value is not None]
    return obj


_SERVICE_FIELDS = sorted(
    (str(field) for field in WorldState.model_fields if field != "meta"),
    key=len,
    reverse=True,
)


def _service_for_name(name: str) -> str | None:
    for field in _SERVICE_FIELDS:
        if name == field or name.startswith(field + "_"):
            return field
    return None


def compute_allowed_services(
    initial_state: dict,
    assertions: list[dict],
    zapier_tools: list[str],
) -> list[str]:
    """Derive the services in scope for one task's simulated workspace."""
    allowed: set[str] = set()
    for key in initial_state:
        if key != "meta" and key in WorldState.model_fields:
            allowed.add(key)
    for assertion in assertions or []:
        service = _service_for_name(str(assertion.get("type", "")))
        if service:
            allowed.add(service)
    for tool_name in zapier_tools or []:
        service = _service_for_name(tool_name)
        if service:
            allowed.add(service)
    return sorted(allowed)
