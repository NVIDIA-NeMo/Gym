# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned trajectory wire formats and their deliberately lossy v1 projection."""

from copy import deepcopy
from typing import Any

from nemo_gym.core.failure_kinds import is_namespaced, is_registered


_TOOL_V2_FIELDS = {
    "agent_instance",
    "span_id",
    "native_tool_call_id",
    "parent_span_id",
    "model_call_id",
    "failure_id",
    "timings",
    "producer_id",
    "model_call_ref",
    "source_message_id",
    "source_model_ref",
}
_TOOL_FLAT_FIELDS = {"started_at", "completed_at", "duration_ms", "timing_source"}


def project_observation_v1(record: dict[str, Any]) -> dict[str, Any]:
    """Project only qualified intervals, keeping unknown values unknown."""
    record = deepcopy(record)
    if record.get("kind") == "tool_call":
        timings = record.get("timings") or []
        # A single explicit boundary is representable by v1's one interval; multiple boundaries
        # require a consumer choice, so the compatibility path does not pick one silently.
        if len(timings) == 1:
            timing = timings[0]
            record.update({key: timing.get(key) for key in ("started_at", "completed_at", "duration_ms")})
            record["timing_source"] = {
                "executor": "executor",
                "lifecycle": "harness",
                "transcript": "artifact",
                "event_receipt": "harness",
            }[timing["boundary"]]
        for key in _TOOL_V2_FIELDS:
            record.pop(key, None)
    else:
        record.pop("agent_instance", None)
    return record


def project_bundle_v1(bundle: dict[str, Any]) -> dict[str, Any]:
    """Return the unversioned observation envelope accepted by existing readers."""
    bundle = deepcopy(bundle)
    bundle.pop("schema_version", None)
    bundle["records"] = [project_observation_v1(record) for record in bundle.get("records", [])]
    for gap in bundle.get("gaps", []):
        gap.pop("agent_instance", None)
        gap.pop("span_id", None)
    return bundle


def project_trajectory_v1(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Return v1, losing attempt, instance, parent, failure and detailed timing provenance.

    This does not establish that v1 consumers can represent retries: consumers that need
    those identities must migrate to v2. No endpoints or durations are manufactured.
    """
    result = deepcopy(trajectory)
    result["schema_version"] = "1.0"
    if "capture_key" in result:
        result["rollout_id"] = result.pop("capture_key")
    result.pop("failure_events", None)
    result["tool_calls"] = [project_observation_v1(record) for record in result.get("tool_calls", [])]
    for turn in result.get("turns", []):
        turn.pop("agent_instance", None)
    for invocation in result.get("invocations", []):
        invocation.pop("agent_instance", None)
    for call in result.get("model_calls", []):
        call.pop("agent_instance", None)
        call.pop("clock_domain", None)
    for gap in result.get("gaps", []):
        gap.pop("agent_instance", None)
        gap.pop("span_id", None)
    return result


def canonical_tool_record(record: dict[str, Any], *, include_join_evidence: bool) -> dict[str, Any]:
    """Keep canonical timing separate from the old flat compatibility projection."""
    result = dict(record)
    for key in _TOOL_FLAT_FIELDS:
        result.pop(key, None)
    if not include_join_evidence:
        for key in ("producer_id", "model_call_ref", "source_message_id", "source_model_ref"):
            result.pop(key, None)
    return result


def validate_failure_events(events: dict[str, dict[str, Any]]) -> None:
    """Validate the shared serialized failure-event map without constructing exceptions."""
    boundaries = {"model", "harness", "agent", "resource", "environment", "collection"}
    allowed = {"boundary", "failure_kind", "failure_reason", "http_status", "foreign_error", "cause_id"}
    for event_id, event in events.items():
        if not event_id or set(event) - allowed:
            raise ValueError("invalid failure event identity or fields")
        if event.get("boundary") not in boundaries:
            raise ValueError("invalid failure boundary")
        if not isinstance(event.get("failure_kind"), str) or not isinstance(event.get("failure_reason"), str):
            raise ValueError("failure kind and reason are required")
        if not (is_registered(event["failure_kind"]) or is_namespaced(event["failure_kind"])):
            raise ValueError("unregistered failure kind")
        if event.get("cause_id") is not None and event["cause_id"] not in events:
            raise ValueError("failure cause must resolve in failure_events")
        if (status := event.get("http_status")) is not None and (type(status) is not int or not 100 <= status <= 599):
            raise ValueError("invalid failure HTTP status")
        foreign = event.get("foreign_error")
        if foreign is not None:
            if not isinstance(foreign, dict) or set(foreign) - {"exception_type", "message", "http_status"}:
                raise ValueError("invalid foreign error details")
            if not all(isinstance(foreign.get(key), str) for key in ("exception_type", "message")):
                raise ValueError("foreign error type and message are required")
        visited = {event_id}
        cause = event.get("cause_id")
        while cause is not None:
            if cause in visited:
                raise ValueError("failure cause references must not form a cycle")
            visited.add(cause)
            cause = events[cause].get("cause_id")


def reject_v2_fields(records: object) -> None:
    """Do not silently discard meaningful v2 evidence mislabeled as a v1 wire record."""
    if not isinstance(records, list):
        return
    for record in records:
        if not isinstance(record, dict):
            continue
        fields = _TOOL_V2_FIELDS - {"model_call_id", "source_message_id", "source_model_ref"}
        if record.get("kind") == "tool_call" or "tool_call_id" in record:
            fields = _TOOL_V2_FIELDS
        if any(record.get(key) not in (None, [], {}) for key in fields):
            raise ValueError("v2 observation fields require schema_version 2.0")
