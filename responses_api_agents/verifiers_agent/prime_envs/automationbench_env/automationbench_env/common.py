"""Shared types and task-scoping helpers for the AutomationBench V1 environment."""

from __future__ import annotations

import verifiers.v1 as vf
from automationbench.schema.world import WorldState

# The single tool server can safely expose the benchmark's conventional bare tool names.
TOOL_PREFIX = None


class AutomationBenchState(vf.State):
    """Per-rollout state bridging the tool server and host-side scoring.

    The rubric runs in the tool server against the live WorldState after every api_fetch;
    only the score crosses the process boundary."""

    partial_credit: float | None = None
    """Rubric partial credit of the current world; None until the agent's first api_fetch,
    in which case scoring falls back to the untouched initial state."""


class AutomationBenchData(vf.TaskData):
    domain: str
    """Upstream domain (sales, marketing, operations, support, finance, hr, simple)."""
    assertions: list[dict]
    """Upstream assertion specs, checked against the final world state by the rubric."""
    initial_state: dict
    """Seed WorldState dict (None-stripped at load, exactly as upstream's setup_state)."""
    zapier_tools: list[str] = []
    """Per-task tool grants, used only to derive the world's connected services."""


class AutomationBenchToolsetConfig(vf.ToolsetConfig):
    pass


def strip_none_values(obj):
    """
    Recursively strip None values from nested dictionaries and lists.

    HuggingFace Dataset normalizes schemas across rows, adding all possible keys
    and setting missing values to None. This breaks Pydantic's default_factory
    since None is passed instead of the field being omitted.
    """
    if isinstance(obj, dict):
        return {k: strip_none_values(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [strip_none_values(item) for item in obj if item is not None]
    else:
        return obj


# Service field names on WorldState, longest first so prefix matching prefers
# "google_sheets" over a hypothetical "google".
_SERVICE_FIELDS = sorted((str(f) for f in WorldState.model_fields if f != "meta"), key=len, reverse=True)


def _service_for_name(name: str) -> str | None:
    """Map an assertion type or tool name to its WorldState service field."""
    for field in _SERVICE_FIELDS:
        field = str(field)
        if name == field or name.startswith(field + "_"):
            return field
    return None


def compute_allowed_services(initial_state: dict, assertions: list[dict], zapier_tools: list[str]) -> list[str]:
    """Derive the set of services a task's world is subscribed to.

    A service is in scope when the task seeds it, asserts on it, or grants one of
    its Zapier tools. Discovery only indexes this set, and execution rejects every
    endpoint outside it.
    """
    allowed: set[str] = set()
    for key in initial_state:
        if key != "meta" and key in WorldState.model_fields:
            allowed.add(key)
    for a in assertions or []:
        service = _service_for_name(str(a.get("type", "")))
        if service:
            allowed.add(service)
    for tool_name in zapier_tools or []:
        service = _service_for_name(tool_name)
        if service:
            allowed.add(service)
    return sorted(allowed)
