# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from nemo_gym.trajectory.trajectory_format import project_bundle_v1, project_trajectory_v1
from nemo_gym.trajectory.trajectory_record import TrajectoryRecord


def test_opencode_projection_preserves_v1_artifact_source_without_relabeling_boundary():
    raw = evidence([("attempt", "native")])
    raw["tool_calls"][0]["timings"] = [
        {
            "boundary": "lifecycle",
            "timing_source": "opencode.tool_part",
            "started_at": 1.0,
            "completed_at": 2.0,
            "duration_ms": 1000.0,
        }
    ]
    projected = project_trajectory_v1(raw)
    assert projected["tool_calls"][0]["timing_source"] == "artifact"
    assert projected["tool_calls"][0]["duration_ms"] == 1000.0
    assert raw["tool_calls"][0]["timings"][0]["boundary"] == "lifecycle"


def evidence(pairs):
    tools = [
        {
            "kind": "tool_call",
            "invocation_id": "root",
            "tool_call_id": old,
            "native_tool_call_id": native,
            "span_id": f"span-{index}",
        }
        for index, (old, native) in enumerate(pairs)
    ]
    calls = [
        {"type": "function_call", "call_id": old, "name": "bash", "arguments": '{"call_id":"literal"}'}
        for old, _ in pairs
    ]
    outputs = [{"type": "function_call_output", "call_id": old, "output": '{"call_id":"literal"}'} for old, _ in pairs]
    invocations = [
        {"kind": "agent_invocation", "invocation_id": "root", "conversation": [*calls, *outputs]},
        {
            "kind": "agent_invocation",
            "invocation_id": "child",
            "parent_invocation_id": "root",
            "spawned_by_tool_call_id": pairs[0][0],
        },
    ]
    turns = [
        {
            "invocation_id": "root",
            "task_id": "task",
            "rollout_id": "capture",
            "turn_no": 1,
            "timestamp": 0,
            "step_count": 0,
            "question": deepcopy(outputs),
            "answer": deepcopy(calls),
        }
    ]
    return {
        "schema_version": "2.0",
        "task_id": "task",
        "capture_key": "capture",
        "tool_calls": tools,
        "invocations": invocations,
        "turns": turns,
    }


def test_native_projection_rewrites_every_reference_atomically_without_touching_payloads():
    raw = evidence([("A", "B"), ("B", "C")])
    raw["model_calls"] = [
        {
            "model_call_id": "model",
            "response": {"output": [{"type": "function_call", "call_id": "A", "arguments": "{}", "name": "bash"}]},
        }
    ]
    before = deepcopy(raw)
    projected = project_trajectory_v1(raw)
    assert raw == before
    assert [t["tool_call_id"] for t in projected["tool_calls"]] == ["B", "C"]
    assert [item["call_id"] for item in projected["invocations"][0]["conversation"]] == ["B", "C", "B", "C"]
    assert [item["call_id"] for item in projected["turns"][0]["question"]] == ["B", "C"]
    assert [item["call_id"] for item in projected["turns"][0]["answer"]] == ["B", "C"]
    assert projected["invocations"][1]["spawned_by_tool_call_id"] == "B"
    assert projected["model_calls"] == raw["model_calls"]
    assert projected["invocations"][0]["conversation"][0]["arguments"] == '{"call_id":"literal"}'
    assert projected["invocations"][0]["conversation"][2]["output"] == '{"call_id":"literal"}'
    TrajectoryRecord.model_validate(projected)
    assert project_trajectory_v1(projected) == projected


@pytest.mark.parametrize("obstacle", ["tool", "reference", "item_id", "source_item_id"])
def test_projection_rejects_collisions_with_unmapped_retained_id(obstacle):
    raw = evidence([("A", "B"), ("B", "C")])
    if obstacle == "tool":
        raw["tool_calls"].append({"invocation_id": "root", "tool_call_id": "C"})
    elif obstacle == "reference":
        raw["turns"][0]["answer"].append({"type": "function_call", "call_id": "C"})
    else:
        raw["invocations"][0]["conversation"].append(
            {"type": "message", "id": "A" if obstacle == "source_item_id" else "C", "role": "assistant", "content": []}
        )
    projected = project_trajectory_v1(raw)
    expected = ["A", "C"] if obstacle == "source_item_id" else ["A", "B"]
    assert [t["tool_call_id"] for t in projected["tool_calls"][:2]] == expected
    assert any(g["code"] == "tool_native_identity_unrepresentable" for g in projected["gaps"])
    assert len(projected["tool_calls"]) == len(raw["tool_calls"])


@pytest.mark.parametrize("same_normalized", [False, True])
def test_reused_native_id_never_merges_distinct_attempts(same_normalized):
    raw = evidence([("A", "native"), ("A" if same_normalized else "B", "native")])
    projected = project_trajectory_v1(raw)
    assert [t["tool_call_id"] for t in projected["tool_calls"]] == ["A", "A" if same_normalized else "B"]
    assert any(g["code"] == "tool_native_identity_unrepresentable" for g in projected["gaps"])


def test_bundle_projection_keeps_spawn_and_compaction_in_owner_scope():
    raw = evidence([("observed", "native")])
    # Same ID in another unique invocation is a separate reference namespace.
    records = [
        *raw["invocations"],
        *raw["tool_calls"],
        {
            "kind": "tool_call",
            "invocation_id": "child",
            "tool_call_id": "observed",
            "native_tool_call_id": "child-native",
            "span_id": "child-span",
        },
        {"kind": "context_compaction", "invocation_id": "root", "first_kept_item_id": "observed"},
    ]
    projected = project_bundle_v1({"schema_version": "2.0", "source": "test", "records": records})
    assert [r["tool_call_id"] for r in projected["records"] if r["kind"] == "tool_call"] == ["native", "child-native"]
    assert projected["records"][1]["spawned_by_tool_call_id"] == "native"
    assert projected["records"][-1]["first_kept_item_id"] == "native"
    assert project_bundle_v1(projected) == projected


def test_existing_v1_record_without_native_metadata_keeps_all_reference_ids():
    raw = evidence([("legacy", "native")])
    raw["schema_version"] = "1.0"
    raw["rollout_id"] = raw.pop("capture_key")
    for tool in raw["tool_calls"]:
        tool.pop("span_id")
        tool.pop("native_tool_call_id")
    assert project_trajectory_v1(raw) == raw


@pytest.mark.parametrize("kind", ["missing_native", "conflicting_native"])
def test_projection_does_not_attribute_ambiguous_source_to_one_native_id(kind):
    raw = evidence([("A", "native")])
    second = {"invocation_id": "root", "tool_call_id": "A", "span_id": "another"}
    if kind == "conflicting_native":
        second["native_tool_call_id"] = "other-native"
    raw["tool_calls"].append(second)
    projected = project_trajectory_v1(raw)
    assert [t["tool_call_id"] for t in projected["tool_calls"]] == ["A", "A"]
    assert projected["gaps"][0]["code"] == "tool_native_identity_unrepresentable"


def test_projection_rewrites_chat_turn_references_without_rewriting_message_ids():
    raw = evidence([("A", "native")])
    raw["turns"][0]["answer"] = {
        "role": "assistant",
        "id": "message",
        "tool_calls": [{"id": "A", "function": {"arguments": '{"call_id":"A"}'}}],
    }
    raw["turns"][0]["question"] = [{"role": "tool", "tool_call_id": "A", "content": "A"}]
    projected = project_trajectory_v1(raw)
    assert projected["turns"][0]["answer"] == {
        "role": "assistant",
        "id": "message",
        "tool_calls": [{"id": "native", "function": {"arguments": '{"call_id":"A"}'}}],
    }
    assert projected["turns"][0]["question"] == [{"role": "tool", "tool_call_id": "native", "content": "A"}]


@pytest.mark.parametrize("explicit_defaults", [False, True])
def test_v1_projection_preserves_sparse_serialization(explicit_defaults):
    raw = {"task_id": "task", "rollout_id": "capture"}
    if explicit_defaults:
        raw.update(schema_version="1.0", tool_calls=[])
    assert project_trajectory_v1(raw) == raw
    record = TrajectoryRecord.model_validate(raw)
    assert record.model_dump(exclude_unset=True) == raw
    assert record.model_dump(mode="json", exclude_unset=True) == raw
    full = record.model_dump()
    assert full["schema_version"] == "1.0"
    assert full["tool_calls"] == []


def test_v1_projection_respects_explicit_field_exclusions():
    record = TrajectoryRecord(task_id="task", rollout_id="capture", schema_version="1.0", tool_calls=[])
    assert record.model_dump(include={"task_id", "rollout_id"}) == {"task_id": "task", "rollout_id": "capture"}
    result = record.model_dump(exclude={"schema_version", "tool_calls"})
    assert "schema_version" not in result
    assert "tool_calls" not in result


@pytest.mark.parametrize("project", [project_bundle_v1, project_trajectory_v1])
def test_v1_drops_only_diagnostics_of_unrepresented_metadata(project):
    metadata_codes = [
        "tool_attempt_identity_unavailable",
        "tool_agent_instance_unavailable",
        "tool_clock_domain_unavailable",
        "tool_duration_source_unavailable",
        "tool_model_call_unavailable",
    ]
    retained_codes = [
        "tool_failure_detail_unavailable",
        "tool_request_payload_unavailable",
        "tool_timing_unavailable",
        "tool_completion_unavailable",
        "agent_transcript_unavailable",
        "model_call_capture_unavailable",
        "producer_trajectory_identity_mismatch",
        "tool_attempt_identity_conflict",
        "tool_native_identity_unrepresentable",
        "tool_output_ambiguous",
        "tool_request_payload_malformed",
        "future_unknown_diagnostic",
    ]
    raw = {"schema_version": "2.0", "gaps": [{"code": code} for code in metadata_codes + retained_codes]}
    before = deepcopy(raw)
    assert [gap["code"] for gap in project(raw)["gaps"]] == retained_codes
    assert raw == before
    assert project(project(raw)) == project(raw)


@pytest.mark.parametrize("project", [project_bundle_v1, project_trajectory_v1])
@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "missing",
        "nonfinite",
        "negative",
        "reversed",
        "changed",
        "ambiguous",
        "malformed",
        "missing_owner",
        "unmatched_owner",
        "invalid_source",
    ],
)
def test_legacy_boundary_gap_requires_preserved_valid_interval(project, invalid):
    import json

    flat = {"started_at": 1.0, "completed_at": 2.0, "duration_ms": 1000.0, "timing_source": "artifact"}
    if invalid == "missing":
        flat.update(started_at=None, completed_at=None, duration_ms=None)
    elif invalid == "nonfinite":
        flat["started_at"] = float("inf")
    elif invalid == "negative":
        flat["duration_ms"] = -1
    elif invalid == "reversed":
        flat["completed_at"] = 0
    elif invalid == "invalid_source":
        flat["timing_source"] = {}
    record = {"kind": "tool_call", "invocation_id": "root", "tool_call_id": "tool", "span_id": "span", **flat}
    gap = {
        "code": "tool_legacy_boundary_unavailable",
        "invocation_id": "root",
        "span_id": "span",
        "detail": json.dumps(flat),
    }
    if invalid == "changed":
        record["duration_ms"] = 2000
    elif invalid == "malformed":
        gap["detail"] = "not json"
    elif invalid == "missing_owner":
        gap.pop("span_id")
    elif invalid == "unmatched_owner":
        gap["span_id"] = "absent"
    records = [record, deepcopy(record)] if invalid == "ambiguous" else [record]
    raw = {
        "schema_version": "2.0",
        "tool_calls": records,
        "records": records,
        "gaps": [gap, {"code": "tool_timing_conflict"}],
    }
    projected = project(raw)
    assert [item["code"] for item in projected["gaps"]] == (
        ["tool_timing_conflict"] if invalid is None else ["tool_legacy_boundary_unavailable", "tool_timing_conflict"]
    )
    assert raw["gaps"][0] == gap
    assert project(projected) == projected
