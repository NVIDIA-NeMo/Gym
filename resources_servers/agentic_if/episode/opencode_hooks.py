# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge the profile plugin's receipts into OpenCode exports and bind planted-event firings to evidence.

The plugin bundle (``opencode_bundle``) stages and compiles the plugins; this module reads what they recorded.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy

from pydantic import JsonValue

from resources_servers.agentic_if.episode.opencode_observations import preserve_rejected_calls
from resources_servers.agentic_if.if_grading.contracts import EventReceipt, SessionEvidence
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec, ToolEvent


def merge_hook_receipts(exports: dict[str, dict], records: list[dict[str, JsonValue]]) -> dict[str, dict]:
    """Enrich a copy of the raw exports using exact invocation identities.

    OpenCode's status/result remains authoritative. Sidecar metadata restores
    facts that stock OpenCode discards for failed calls; it never changes an
    error into a success. Raw exports are retained separately for auditing.
    """
    result = deepcopy(exports)
    parts = {}
    for session, document in result.items():
        for message in document.get("messages", []):
            for part in message.get("parts", []):
                if part.get("type") != "tool":
                    continue
                identity = (session, part.get("callID"))
                if identity in parts:
                    raise ValueError("ambiguous exported invocation")
                parts[identity] = part
    seen = set()
    for record in records:
        if record.get("kind") != "tool_receipt":
            continue
        identity = (record.get("session_id"), record.get("call_id"))
        part = parts.get(identity)
        if identity in seen or part is None or part.get("tool") != record.get("opencode_tool"):
            raise ValueError("unbound or duplicate hook execution receipt")
        seen.add(identity)
        state = part["state"]
        stamp = record["metadata"]
        state.setdefault("metadata", {})["nemo_gym_profile"] = deepcopy(stamp)
        if "configured_args" in stamp:
            state["input"] = deepcopy(stamp["configured_args"])
    presented = {}
    for record in records:
        if record.get("kind") != "tool_presentation":
            continue
        identity = (record.get("session_id"), record.get("call_id"))
        part = parts.get(identity)
        state = part.get("state", {}) if part else {}
        if (
            (identity in presented and presented[identity] != record.get("model_error"))
            or part is None
            or part.get("tool") != record.get("opencode_tool")
            or state.get("status") != "error"
            or not isinstance(record.get("opencode_error"), str)
            or record["opencode_error"] != state.get("error")
            or not isinstance(record.get("model_error"), str)
        ):
            raise ValueError("unbound or duplicate hook presentation receipt")
        stamp = state.setdefault("metadata", {}).setdefault("nemo_gym_profile", {})
        if "model_error" in stamp and stamp["model_error"] != record["model_error"]:
            raise ValueError("hook presentation disagrees with execution receipt")
        stamp["model_error"] = record["model_error"]
        # Replaying history may present the same error in another OpenCode
        # process. Identical receipts are idempotent; conflicting views fail.
        presented[identity] = record["model_error"]
    return result


def enrich_exports(
    exports: dict[str, dict],
    records: list[dict[str, JsonValue]],
    *,
    spec: RuntimeSpec,
    observations: Sequence[Mapping[str, JsonValue]],
) -> dict[str, dict]:
    """Receipt-enriched copies of the OpenCode exports; the raw exports stay unchanged for auditing."""
    enriched = merge_hook_receipts(exports, records)
    if spec.tools.tool_names is not None:
        preserve_rejected_calls(enriched, observations, spec.continuation)
    return enriched


def attach_events(
    evidence: SessionEvidence, events: list[ToolEvent], records: list[dict[str, JsonValue]]
) -> SessionEvidence:
    """Bind actual plant firings to captured tool results; never infer a firing."""
    result = evidence.model_copy(deep=True)
    issues = []
    if not any(row.get("kind") == "loaded" and row.get("plants") == len(events) for row in records):
        issues.append("missing_event_load_receipt")
    declared = {event.id: event for event in events}
    seen = {event.id for event in result.events}
    for record in records:
        if record.get("kind") == "error":
            issues.append("event_load_error")
        if record.get("kind") != "fired":
            continue
        identity = record.get("event_id")
        event = declared.get(identity) if isinstance(identity, str) else None
        if event is None or identity in seen:
            issues.append("unknown_or_duplicate_event")
            continue
        seen.add(identity)
        matches = [
            index
            for index, step in enumerate(result.steps)
            if step.role == "tool"
            and step.session_id == record.get("session_id")
            and step.call_id == record.get("call_id")
            and step.tool == event.match.tool
        ]
        if len(matches) != 1 or record.get("action") != event.action or record.get("tool") != event.match.tool:
            issues.append(f"uncorrelated_event:{identity}")
            continue
        result.events.append(
            EventReceipt(
                id=identity,
                step_index=matches[0],
                call_id=record["call_id"],
                invocation_id=result.steps[matches[0]].invocation_id,
                details=record,
            )
        )
        result.steps[matches[0]].planted = True
    if issues:
        result.status = "incomplete"
        result.issues.extend(issues)
    return result


def check_hook_receipts(
    evidence: SessionEvidence,
    *,
    records: list[dict[str, JsonValue]],
    profile_sha256: str | None,
    events: list[ToolEvent],
    event_records: list[dict[str, JsonValue]],
) -> SessionEvidence:
    """Missing/malformed receipts make evidence incomplete, never a silent no-op.

    The profile plugin fails closed: without its load receipt for the staged profile, nothing it was meant to apply
    can be trusted.
    """
    result = evidence.model_copy(deep=True)
    try:
        if not profile_sha256 or not any(
            row.get("kind") == "loaded" and row.get("profile_sha256") == profile_sha256 for row in records
        ):
            raise ValueError("missing live hook load receipt")
        if events:
            result = attach_events(result, events, event_records)
    except (ValueError, TypeError) as error:
        result.status = "incomplete"
        result.issues.append(f"hook_receipt_error:{error}")
    return result
