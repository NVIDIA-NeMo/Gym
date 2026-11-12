# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned trajectory wire formats and their deliberately lossy v1 projection."""

import json
import math
from collections import defaultdict
from copy import deepcopy
from typing import Any

from nemo_gym.core.failure_records import FailureReport
from nemo_gym.trajectory.model_call_record import tool_request_evidence


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

# These diagnose only metadata that v1 cannot represent. Coverage, conflicts and unknown
# diagnostics remain visible to legacy consumers, including strict exporters.
_V2_ONLY_GAPS = frozenset(
    {
        "failure_report_unavailable",
        "tool_attempt_identity_unavailable",
        "tool_agent_instance_unavailable",
        "tool_clock_domain_unavailable",
        "tool_duration_source_unavailable",
        "tool_model_call_unavailable",
    }
)


def _legacy_interval_preserved(gap: dict[str, Any], tools: list[dict[str, Any]]) -> bool:
    if gap.get("code") != "tool_legacy_boundary_unavailable":
        return False
    if not all(isinstance(gap.get(key), str) and gap[key] for key in ("invocation_id", "span_id")):
        return False
    try:
        original = json.loads(gap.get("detail"))
    except (TypeError, ValueError):
        return False
    if not isinstance(original, dict) or set(original) != _TOOL_FLAT_FIELDS:
        return False
    values = [original[key] for key in ("started_at", "completed_at", "duration_ms")]
    if all(value is None for value in values) or any(
        value is not None and (type(value) not in {int, float} or not math.isfinite(value)) for value in values
    ):
        return False
    start, end, duration = values
    if (duration is not None and duration < 0) or (start is not None and end is not None and end < start):
        return False
    if original["timing_source"] not in (None, "executor", "artifact", "harness"):
        return False
    matching = [
        tool
        for tool in tools
        if all(
            gap.get(key) is None or tool.get(key) == gap[key] for key in ("invocation_id", "agent_instance", "span_id")
        )
    ]
    if len(matching) != 1:
        return False
    projected = project_observation_v1(matching[0])
    return all(projected.get(key) == value for key, value in original.items())


def _project_gaps_v1(gaps: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in gap.items() if key not in {"agent_instance", "span_id"}}
        for gap in gaps
        if gap.get("code") not in _V2_ONLY_GAPS and not _legacy_interval_preserved(gap, tools)
    ]


def project_observation_v1(record: dict[str, Any]) -> dict[str, Any]:
    """Project only qualified intervals, keeping unknown values unknown."""
    record = deepcopy(record)
    if record.get("kind", "tool_call" if "tool_call_id" in record else None) == "tool_call":
        timings = record.get("timings") or []
        # A single explicit boundary is representable by v1's one interval; multiple boundaries
        # require a consumer choice, so the compatibility path does not pick one silently.
        if len(timings) == 1:
            timing = timings[0]
            record.update({key: timing.get(key) for key in ("started_at", "completed_at", "duration_ms")})
            # V1's source labels mix delivery and measurement semantics. Preserve the
            # established OpenCode artifact label without changing its v2 boundary.
            record["timing_source"] = (
                "artifact"
                if timing.get("timing_source") == "opencode.tool_part"
                else {
                    "executor": "executor",
                    "lifecycle": "harness",
                    "transcript": "artifact",
                    "event_receipt": "harness",
                }[timing["boundary"]]
            )
        elif len(timings) > 1:
            record.update({key: None for key in _TOOL_FLAT_FIELDS})
        for key in _TOOL_V2_FIELDS:
            record.pop(key, None)
    else:
        record.pop("agent_instance", None)
    return record


def _restore_native_tool_ids(
    tool_groups: list[list[dict[str, Any]]],
    invocations: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    compactions: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    model_calls: list[dict[str, Any]],
) -> None:
    """Restore v1's native request namespace only with an explicit, collision-free mapping.

    Model-call payloads and text are independent evidence and are never rewritten. All
    reference values are collected before any edit so chained mappings cannot cascade.
    """
    tools = [tool for group in tool_groups for tool in group]
    edges = defaultdict(lambda: defaultdict(set))
    attempts = defaultdict(lambda: defaultdict(set))
    source_attempts = defaultdict(lambda: defaultdict(set))
    slots = []
    retained = defaultdict(set)
    immutable = defaultdict(set)

    def references(value):
        """Yield typed request/result references and their matching item-ID aliases."""
        for item in value if isinstance(value, list) else [value]:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind in {"function_call", "custom_tool_call", "function_call_output", "custom_tool_call_output"}:
                role = "output" if kind.endswith("_output") else "request"
                yield item, "call_id", role
                if item.get("id") is not None and item.get("id") == item.get("call_id"):
                    yield item, "id", role
            elif item.get("role") == "tool":
                yield item, "tool_call_id", "output"
            elif item.get("role") == "assistant" and isinstance(item.get("tool_calls"), list):
                for call in item["tool_calls"]:
                    if isinstance(call, dict):
                        yield call, "id", "request"

    def content(payload, *, response):
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                return None
        if not isinstance(payload, dict):
            return None
        if not response:
            return payload.get("input", payload.get("messages"))
        if isinstance(payload.get("output"), list):
            return payload["output"]
        choices = payload.get("choices")
        if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
            return choices[0].get("message")
        return None

    calls_by_id = defaultdict(list)
    for call in model_calls:
        if call.get("model_call_id"):
            calls_by_id[call["model_call_id"]].append(call)
    # Only a unique exact requesting call can establish already-native aliases.
    native_requests = defaultdict(set)
    for tool in tools:
        matches = calls_by_id.get(tool.get("model_call_id"), [])
        if len(matches) != 1:
            continue
        call = matches[0]
        metadata = call.get("response_metadata") or {}
        evidence = tool_request_evidence(
            call.get("response"),
            status_code=metadata.get("status_code"),
            response_status=metadata.get("response_status"),
        )
        native = tool.get("native_tool_call_id")
        request_ids = [
            record.get(field)
            for record, field, role in references(content(call.get("response"), response=True))
            if role == "request" and not (field == "id" and "call_id" in record)
        ]
        if evidence.kind == "available" and native in evidence.call_ids and request_ids.count(native) == 1:
            native_requests[tool.get("invocation_id"), native].add(call["model_call_id"])

    def reference(record, field, owner, *, proven_native=False):
        value = record.get(field)
        if isinstance(owner, str) and isinstance(value, str) and not proven_native:
            slots.append((record, field, owner, value))
            retained[owner].add(value)

    def items(value, owner, *, native_aliases=()):
        typed = list(references(value))
        aliases = {
            id(record) for record, field, _ in typed if field == "id" and record.get("id") == record.get("call_id")
        }
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, dict) and isinstance(item.get("id"), str) and id(item) not in aliases:
                immutable[owner].add(item["id"])
        for record, field, role in typed:
            reference(record, field, owner, proven_native=(role, record.get(field)) in native_aliases)

    def turn_aliases(turn, field):
        owner = turn.get("invocation_id")
        refs = turn.get("model_calls") or []
        if len(refs) != 1 or not isinstance(refs[0], dict):
            return set()
        matches = calls_by_id.get(refs[0].get("model_call_id"), [])
        if len(matches) != 1:
            return set()
        call = matches[0]
        metadata = call.get("response_metadata") or {}
        if any(
            refs[0].get(key) is not None and refs[0][key] != metadata.get(key) for key in ("response_id", "model_ref")
        ):
            return set()
        payload = content(call.get("response" if field == "answer" else "request"), response=field == "answer")
        result = set()
        for record, key, role in references(payload):
            native = record.get(key)
            requesting = native_requests.get((owner, native), set())
            if len(requesting) == 1 and (field == "question" or requesting == {call["model_call_id"]}):
                result.add((role, native))
        return result

    # Copies in different envelopes need the same mapping, but do not constitute
    # additional attempts. Ambiguity within either envelope still blocks it.
    for group_index, group in enumerate(tool_groups):
        for tool in group:
            owner, old, native = tool.get("invocation_id"), tool.get("tool_call_id"), tool.get("native_tool_call_id")
            reference(tool, "tool_call_id", owner)
            source_attempts[owner, old][group_index].add(tool.get("span_id") or ("legacy", old))
            if isinstance(owner, str) and isinstance(old, str) and isinstance(native, str) and native:
                edges[owner][old].add(native)
                attempts[owner, native][group_index].add(tool.get("span_id") or ("legacy", old))
    for invocation in invocations:
        owner = invocation.get("invocation_id")
        items(invocation.get("conversation", []), owner)
        reference(invocation, "spawned_by_tool_call_id", invocation.get("parent_invocation_id"))
    for turn in turns:
        for field in ("question", "answer"):
            items(turn.get(field), turn.get("invocation_id"), native_aliases=turn_aliases(turn, field))
    for compaction in compactions:
        reference(compaction, "first_kept_item_id", compaction.get("invocation_id"))

    mappings = {}
    for owner, requests in edges.items():
        incoming = defaultdict(set)
        for old, natives in requests.items():
            for native in natives:
                incoming[native].add(old)
        mapping = {
            old: next(iter(natives))
            for old, natives in requests.items()
            if len(natives) == 1
            and all(len(group) == 1 for group in source_attempts[owner, old].values())
            and len(incoming[next(iter(natives))]) == 1
            and all(len(group) == 1 for group in attempts[owner, next(iter(natives))].values())
        }
        # A destination may reuse an ID being renamed, but never an ID which stays.
        # Removing one unsafe edge can expose another collision, hence the fixed point.
        while True:
            blocked = {
                old
                for old, native in mapping.items()
                if old != native
                and (
                    old in immutable[owner] or native in immutable[owner] or native in retained[owner] - mapping.keys()
                )
            }
            if not blocked:
                break
            mapping = {old: native for old, native in mapping.items() if old not in blocked}
        mappings[owner] = mapping
        for old in requests.keys() - mapping.keys():
            gap = {
                "code": "tool_native_identity_unrepresentable",
                "invocation_id": owner,
                "detail": json.dumps({"tool_call_id": old, "reason": "ambiguous_or_colliding_native_id"}),
            }
            if gap not in gaps:
                gaps.append(gap)
    for record, field, owner, old in slots:
        record[field] = mappings.get(owner, {}).get(old, old)


def project_bundle_v1(bundle: dict[str, Any], *, restore_native_ids: bool = True) -> dict[str, Any]:
    """Return the unversioned observation envelope accepted by existing readers."""
    bundle = deepcopy(bundle)
    records = bundle.get("records", [])
    gaps = bundle.get("gaps", [])
    if restore_native_ids:
        _restore_native_tool_ids(
            [[r for r in records if r.get("kind") == "tool_call"]],
            [r for r in records if r.get("kind") == "agent_invocation"],
            [],
            [r for r in records if r.get("kind") == "context_compaction"],
            gaps,
            [],
        )
    if gaps:
        bundle["gaps"] = gaps
    bundle.pop("schema_version", None)
    if "records" in bundle:
        bundle["records"] = [project_observation_v1(record) for record in bundle["records"]]
    if "gaps" in bundle:
        bundle["gaps"] = _project_gaps_v1(
            bundle["gaps"], [record for record in records if record.get("kind") == "tool_call"]
        )
    return bundle


def project_trajectory_v1(trajectory: dict[str, Any], *, restore_native_ids: bool = True) -> dict[str, Any]:
    """Return v1, losing attempt, instance, parent, failure and detailed timing provenance.

    This does not establish that v1 consumers can represent retries: consumers that need
    those identities must migrate to v2. No endpoints or durations are manufactured.
    """
    result = deepcopy(trajectory)
    gaps = result.get("gaps", [])
    if restore_native_ids:
        _restore_native_tool_ids(
            [result.get("tool_calls", [])],
            [*result.get("invocations", []), *result.get("verifier_invocations", [])],
            result.get("turns", []),
            [],
            gaps,
            result.get("model_calls", []),
        )
    if gaps:
        result["gaps"] = gaps
    if "schema_version" in result:
        result["schema_version"] = "1.0"
    if "capture_key" in result:
        result["rollout_id"] = result.pop("capture_key")
    result.pop("failure_events", None)
    if "tool_calls" in result:
        result["tool_calls"] = [project_observation_v1(record) for record in result["tool_calls"]]
    for turn in result.get("turns", []):
        turn.pop("agent_instance", None)
    for invocation in result.get("invocations", []):
        invocation.pop("agent_instance", None)
    for call in result.get("model_calls", []):
        call.pop("agent_instance", None)
        call.pop("clock_domain", None)
    if "gaps" in result:
        result["gaps"] = _project_gaps_v1(result["gaps"], trajectory.get("tool_calls", []))
    return result


def project_result_v1(
    trajectory: dict[str, Any], bundle: dict[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Export both retained envelopes with one atomic, conservative identity decision."""
    trajectory, bundle = deepcopy(trajectory), deepcopy(bundle)
    records = bundle.get("records", []) if bundle is not None else []
    gaps: list[dict[str, Any]] = []
    _restore_native_tool_ids(
        [trajectory.get("tool_calls", []), [r for r in records if r.get("kind") == "tool_call"]],
        [
            *trajectory.get("invocations", []),
            *trajectory.get("verifier_invocations", []),
            *[r for r in records if r.get("kind") == "agent_invocation"],
        ],
        trajectory.get("turns", []),
        [r for r in records if r.get("kind") == "context_compaction"],
        gaps,
        trajectory.get("model_calls", []),
    )
    for envelope in (trajectory, bundle):
        if envelope is not None and gaps:
            retained = envelope.setdefault("gaps", [])
            retained.extend(gap for gap in gaps if gap not in retained)
    return (
        project_trajectory_v1(trajectory, restore_native_ids=False),
        project_bundle_v1(bundle, restore_native_ids=False) if bundle is not None else None,
    )


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
    FailureReport(events=events)


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
        if record.get("kind") == "sandbox":
            # Sandbox outcomes already carried failure references in schema v1.
            fields = fields - {"failure_id"}
        if any(record.get(key) not in (None, [], {}) for key in fields):
            raise ValueError("v2 observation fields require schema_version 2.0")
