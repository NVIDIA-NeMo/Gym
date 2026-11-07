# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_gym.core.operation_timing import OperationTiming
from nemo_gym.trajectory.tool_timing import normalize_operation_timing, normalize_tool_calls
from nemo_gym.trajectory.trajectory_format import project_trajectory_v1
from nemo_gym.trajectory.trajectory_record import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
    TrajectoryModelCall,
    TrajectoryRecord,
)


def timing(**updates):
    return OperationTiming(
        boundary="lifecycle",
        timing_source="test.tool",
        started_at=10,
        completed_at=12,
        duration_ms=2000,
        duration_source="monotonic",
        **updates,
    )


def call(identifier="model", *, instance="assistant", native_id="native", message="message"):
    return TrajectoryModelCall(
        model_call_id=identifier,
        agent_instance=instance,
        client_session_id="root",
        client_assistant_message_id=message,
        response={"output": [{"type": "function_call", "call_id": native_id, "name": "bash", "arguments": "{}"}]},
    )


def tool(span="attempt", *, instance="assistant", **updates):
    return ToolCallObservation(
        invocation_id="root",
        agent_instance=instance,
        span_id=span,
        tool_call_id="normalized",
        native_tool_call_id="native",
        producer_id="producer",
        source_message_id="message",
        timings=[timing()],
        **updates,
    )


def normalize(tools, *, calls=None, invocations=None, gaps=()):
    return normalize_tool_calls(
        tools,
        capture_key="0-0-a1",
        producer_id="test",
        model_calls=calls or [call()],
        invocations=invocations or [AgentInvocation(invocation_id="root", agent_instance="assistant")],
        observation_gaps=gaps,
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_timing_rejects_nonfinite_measurements(value):
    with pytest.raises(ValidationError):
        OperationTiming(boundary="lifecycle", timing_source="test", duration_ms=value)


def test_timing_salvages_monotonic_duration_without_fabricating_end():
    measured, gaps = normalize_operation_timing(
        boundary="executor",
        timing_source="test",
        started_at=10,
        completed_at=9,
        duration_ms=0,
        duration_source="monotonic",
    )
    assert measured.started_at == 10
    assert measured.completed_at is None
    assert measured.duration_ms == 0
    assert "tool_clock_discontinuity" in gaps
    assert normalize_operation_timing(boundary="lifecycle", timing_source="test")[0] is None
    with pytest.raises(ValidationError):
        OperationTiming(boundary="lifecycle", timing_source="test")


def test_exact_message_join_disambiguates_reused_native_id():
    tools, gaps = normalize([tool()], calls=[call(), call("other", message="other")])
    assert tools[0].model_call_id == "model"
    assert not any(g.code.startswith("tool_model_call") for g in gaps)
    tools, gaps = normalize([tool()], calls=[call(), call("retry")])
    assert tools[0].model_call_id is None
    assert "tool_model_call_ambiguous" in {g.code for g in gaps}


def test_unknown_source_never_joins_by_native_call_id_alone():
    observed = tool().model_copy(update={"source_message_id": None})
    tools, gaps = normalize([observed])
    assert tools[0].model_call_id is None
    assert "tool_model_call_unavailable" in {g.code for g in gaps}


def test_retries_remain_distinct_and_replay_deduplicates():
    observed = [tool("first", status="failed"), tool("second", status="completed")]
    records, gaps = normalize([*observed, *observed])
    assert len({t.span_id for t in records}) == 2
    assert [(t.status, t.model_call_id) for t in records] == [("failed", "model"), ("completed", "model")]
    assert [t.span_id for t in normalize(records)[0]] == [t.span_id for t in records]
    assert "tool_failure_detail_unavailable" in {g.code for g in gaps}


def test_instance_resolution_precedes_attempt_merge():
    records, _ = normalize([tool(instance=None), tool()])
    assert len(records) == 1
    assert records[0].agent_instance == "assistant"


def test_distinct_instances_and_unattributed_producers_do_not_merge():
    with pytest.raises(ValueError, match="duplicate IDs across agent instances are unsupported"):
        normalize(
            [tool(instance="a"), tool(instance="b")],
            invocations=[AgentInvocation(invocation_id="root", agent_instance=x) for x in ("a", "b")],
            calls=[call("a", instance="a"), call("b", instance="b")],
        )
    unknown = [tool(instance=None), tool(instance=None).model_copy(update={"producer_id": "another"})]
    records, _ = normalize(unknown, invocations=[AgentInvocation(invocation_id="root")], calls=[call(instance=None)])
    assert len(records) == 2
    assert len({t.span_id for t in records}) == 2
    assert all(t.agent_instance is None for t in records)


def test_unknown_instance_is_not_a_wildcard_for_model_ownership():
    records, gaps = normalize(
        [tool(instance=None)],
        invocations=[AgentInvocation(invocation_id="root")],
        calls=[call("a", instance="a"), call("b", instance="b")],
    )
    assert records[0].model_call_id is None
    assert "tool_model_call_ambiguous" in {g.code for g in gaps}


def test_v2_decoder_checks_request_and_failure_references():
    records, _ = normalize([tool(model_call_ref=ModelCallRef(model_call_id="model"))])
    trajectory = TrajectoryRecord(
        schema_version="2.0", task_id="0", capture_key="0-0-a1", model_calls=[call()], tool_calls=records
    )
    data = trajectory.model_dump(mode="json")
    assert data["capture_key"] == "0-0-a1" and "rollout_id" not in data
    assert "started_at" not in data["tool_calls"][0]
    assert "producer_id" not in data["tool_calls"][0]
    assert project_trajectory_v1(data)["tool_calls"][0]["duration_ms"] == 2000
    for changed in (
        {"rollout_id": "conflict"},
        {"model_calls": [call(native_id="wrong").model_dump()]},
        {"tool_calls": [{**data["tool_calls"][0], "failure_id": "missing"}]},
        {"tool_calls": [*data["tool_calls"], *data["tool_calls"]]},
    ):
        with pytest.raises(ValidationError):
            TrajectoryRecord.model_validate(data | changed)
    with pytest.raises(ValidationError):
        TrajectoryRecord.model_validate(data | {"schema_version": "9.0"})
    with pytest.raises(ValidationError):
        TrajectoryRecord.model_validate({**data, "schema_version": "1.0", "rollout_id": data["capture_key"]})


def test_model_attribution_resolution_deduplicates_existing_attempt():
    unknown = tool(instance=None, model_call_id="model")
    known = tool(model_call_id="model")
    attempts, _ = normalize([unknown, known], invocations=[AgentInvocation(invocation_id="root")])
    assert len(attempts) == 1
    assert attempts[0].agent_instance == "assistant"


@pytest.mark.parametrize("record_kind", ["tool", "model"])
def test_v2_decoder_rejects_known_invocation_instance_mismatch(record_kind):
    invocation = AgentInvocation(invocation_id="root", agent_instance="other")
    if record_kind == "model":
        invocation.model_calls = [ModelCallRef(model_call_id="model")]
    with pytest.raises(ValidationError, match="owning invocation"):
        TrajectoryRecord(
            schema_version="2.0",
            task_id="task",
            rollout_id="capture",
            invocations=[invocation],
            model_calls=[call()],
            tool_calls=[tool(model_call_id="model").model_dump()] if record_kind == "tool" else [],
        )


def test_producer_span_mapping_is_stable_before_collision_and_translates_parents():
    def producer_tools(producer):
        return [
            tool(span, instance=None).model_copy(update={"producer_id": producer, "parent_span_id": parent})
            for span, parent in (("parent", None), ("child", "parent"))
        ]

    first, _ = normalize(
        producer_tools("one"), calls=[call(instance=None)], invocations=[AgentInvocation(invocation_id="root")]
    )
    later, _ = normalize(
        [*first, *producer_tools("one"), *producer_tools("two")],
        calls=[call(instance=None)],
        invocations=[AgentInvocation(invocation_id="root")],
    )
    assert len(later) == 4
    assert [t.span_id for t in later[:2]] == [t.span_id for t in first]
    assert len({t.span_id for t in later}) == 4
    for producer in ("one", "two"):
        parent, child = [t for t in later if t.producer_id == producer]
        assert child.parent_span_id == parent.span_id
    decoded = TrajectoryRecord.model_validate(
        TrajectoryRecord(
            schema_version="2.0",
            task_id="task",
            rollout_id="capture",
            tool_calls=later,
            model_calls=[call(instance=None)],
        ).model_dump(mode="json")
    )
    replay, _ = normalize(
        [*decoded.tool_calls, *producer_tools("one"), *producer_tools("two")],
        calls=[call(instance=None)],
        invocations=[AgentInvocation(invocation_id="root")],
    )
    assert len(replay) == 4
    assert [(t.span_id, t.parent_span_id) for t in replay] == [(t.span_id, t.parent_span_id) for t in later]


def assert_gap_targets(tools, gaps):
    for gap in gaps:
        if gap.span_id is not None:
            targets = [item for item in tools if gap.matches_tool(item)]
            assert len(targets) == 1
            assert (gap.agent_instance, gap.invocation_id) == (targets[0].agent_instance, targets[0].invocation_id)


@pytest.mark.parametrize("instances", [(None, None)])
def test_gap_import_resolves_unique_owner_and_never_fans_out_ambiguous_spans(instances):
    observations = [
        tool("shared", instance=instance).model_copy(update={"producer_id": producer})
        for producer, instance in zip(("one", "two"), instances)
    ]
    raw_gaps = [ObservationGap(code="adapter_only", invocation_id="root", span_id="shared", detail="kept")]
    if instances[0] is not None:
        raw_gaps.append(
            ObservationGap(code="owned_adapter", invocation_id="root", agent_instance="a", span_id="shared")
        )
    invocations = [AgentInvocation(invocation_id="root", agent_instance=instance) for instance in set(instances)]
    tools, gaps = normalize(observations, invocations=invocations, calls=[call(instance=None)], gaps=raw_gaps)
    assert len(tools) == 2
    (unresolved,) = [gap for gap in gaps if gap.code == "adapter_only"]
    assert unresolved.span_id is None
    assert json.loads(unresolved.detail) == {
        "unresolved_span_reference": {"span_id": "shared", "reason": "ambiguous"},
        "detail": "kept",
    }
    if instances[0] is not None:
        (owned,) = [gap for gap in gaps if gap.code == "owned_adapter"]
        assert owned.span_id == next(tool.span_id for tool in tools if tool.agent_instance == "a")
        assert owned.agent_instance == "a"
    assert_gap_targets(tools, gaps)


@pytest.mark.parametrize("resolve_through_model", [False, True])
def test_imported_and_generated_gap_owners_follow_exact_resolution(resolve_through_model):
    observed = tool(instance=None, model_call_id="model")
    raw = ObservationGap(code="adapter_clock_invalid", invocation_id="root", span_id="attempt")
    owner = None if resolve_through_model else "assistant"
    tools, gaps = normalize(
        [observed], invocations=[AgentInvocation(invocation_id="root", agent_instance=owner)], gaps=[raw]
    )
    (imported,) = [gap for gap in gaps if gap.code == raw.code]
    assert imported.span_id == tools[0].span_id
    assert imported.agent_instance == "assistant"
    assert_gap_targets(tools, gaps)


def test_legacy_diagnostics_have_final_attempt_scope():
    raw = ToolCallObservation(
        invocation_id="root", tool_call_id="request", started_at=1, completed_at=2, timing_source="artifact"
    )
    tools, gaps = normalize([raw])
    (diagnostic,) = [gap for gap in gaps if gap.code == "tool_legacy_boundary_unavailable"]
    assert diagnostic.span_id == tools[0].span_id
    assert diagnostic.agent_instance == tools[0].agent_instance == "assistant"
    assert_gap_targets(tools, gaps)


@pytest.mark.parametrize(
    "gap",
    [
        ObservationGap(code="bad", span_id="missing"),
        ObservationGap(code="bad", span_id="attempt", invocation_id="wrong"),
        ObservationGap(code="bad", span_id="attempt", agent_instance="wrong"),
    ],
)
def test_canonical_decoder_rejects_dangling_or_misowned_gap(gap):
    with pytest.raises(ValidationError, match="attempt-scoped gap"):
        TrajectoryRecord(
            schema_version="2.0", task_id="task", rollout_id="capture", tool_calls=[tool().model_dump()], gaps=[gap]
        )


def test_unmatched_raw_gap_stays_explicitly_unresolved_without_touching_other_namespaces():
    observed = tool(model_call_id="model", failure_id="failure").model_copy(update={"status": "failed"})
    tools, gaps = normalize_tool_calls(
        [observed],
        capture_key="capture",
        producer_id="producer",
        model_calls=[call()],
        invocations=[AgentInvocation(invocation_id="root", agent_instance="assistant")],
        failure_events={
            "failure": {"boundary": "harness", "failure_kind": "example:failure", "failure_reason": "failed"}
        },
        observation_gaps=[ObservationGap(code="native", span_id="missing", detail="native evidence")],
    )
    assert tools[0].model_call_id == "model"
    assert tools[0].failure_id == "failure"
    assert tools[0].tool_call_id == "normalized"
    assert tools[0].native_tool_call_id == "native"
    (unresolved,) = [gap for gap in gaps if gap.code == "native"]
    assert unresolved.span_id is None
    assert json.loads(unresolved.detail)["unresolved_span_reference"] == {"span_id": "missing", "reason": "unmatched"}
    assert_gap_targets(tools, gaps)


def test_one_import_maps_parent_child_and_their_diagnostics_together():
    parent = tool("parent", instance=None)
    child = tool("child", instance=None, parent_span_id="parent")
    raw_gaps = [
        ObservationGap(code="adapter_only", span_id=span, invocation_id="root") for span in ("parent", "child")
    ]
    tools, gaps = normalize([parent, child], gaps=raw_gaps)
    assert tools[1].parent_span_id == tools[0].span_id
    imported = [gap for gap in gaps if gap.code == "adapter_only"]
    assert [gap.span_id for gap in imported] == [item.span_id for item in tools]
    assert all(gap.agent_instance == "assistant" for gap in imported)
    assert_gap_targets(tools, gaps)


@pytest.mark.parametrize("servers, expected", [(["b"], None), (["a", "b"], "a"), (["a"], "a")])
def test_replay_preserves_every_request_qualifier(servers, expected):
    from nemo_gym.config.config_types import ModelServerRef

    observed = tool(source_model_ref=ModelServerRef(type="responses_api_models", name="a"))
    initial, _ = normalize_tool_calls(
        [observed],
        capture_key="0-0-a1",
        producer_id="test",
        model_calls=[],
        invocations=[AgentInvocation(invocation_id="root", agent_instance="assistant")],
    )
    saved = TrajectoryRecord(schema_version="2.0", task_id="task", capture_key="0-0-a1", tool_calls=initial)
    captured = []
    for server in servers:
        captured_call = call(server)
        captured_call.response_metadata.model_ref = ModelServerRef(type="responses_api_models", name=server)
        captured.append(captured_call)
    for _ in range(3):
        tools, _ = normalize(
            [*TrajectoryRecord.model_validate(saved.model_dump(mode="json")).tool_calls, observed], calls=captured
        )
        assert tools[0].model_call_id == expected
        assert tools[0].span_id == initial[0].span_id
        saved = TrajectoryRecord(
            schema_version="2.0", task_id="task", capture_key="0-0-a1", tool_calls=tools, model_calls=captured
        )


@pytest.mark.parametrize("contradiction", ["response_id", "model_ref", "model_call_id", "snapshot"])
def test_request_reference_contradictions_never_link(contradiction):
    from nemo_gym.config.config_types import ModelServerRef

    captured = call()
    captured.response_metadata.response_id = "response"
    captured.response_metadata.model_ref = ModelServerRef(type="responses_api_models", name="policy")
    reference = ModelCallRef(
        model_call_id="model", response_id="response", model_ref=captured.response_metadata.model_ref
    )
    observed = tool(model_call_id="model", model_call_ref=reference)
    if contradiction == "snapshot":
        other = observed.model_copy(update={"model_call_ref": reference.model_copy(update={"response_id": "other"})})
        observations = [observed, other]
    else:
        update = ModelServerRef(type="responses_api_models", name="other") if contradiction == "model_ref" else "other"
        observed.model_call_ref = reference.model_copy(update={contradiction: update})
        observations = [observed]
    tools, gaps = normalize(observations, calls=[captured])
    assert tools[0].model_call_id is None
    assert any(g.code.startswith("tool_model_call") for g in gaps)


@pytest.mark.parametrize(
    "first, second, expected",
    [
        ("unknown", "incomplete", "incomplete"),
        ("incomplete", "unknown", "incomplete"),
        ("incomplete", "failed", "failed"),
        ("incomplete", "completed", "completed"),
        ("completed", "unknown", "completed"),
        ("completed", "failed", "completed"),
    ],
)
def test_snapshot_outcomes_are_monotonic(first, second, expected):
    tools, gaps = normalize(
        [tool(status=first), tool(status=second, error_type="exit_7" if second == "failed" else None)]
    )
    assert tools[0].status == expected
    assert tools[0].error_type == ("exit_7" if expected == second == "failed" else None)
    assert ("tool_completion_unavailable" in {g.code for g in gaps}) == (expected == "incomplete")
    assert ("tool_outcome_conflict" in {g.code for g in gaps}) == (first == "completed" and second == "failed")
    assert normalize([tools[0], tool(status="unknown")])[0][0].error_type == tools[0].error_type


def test_raw_default_tool_kind_projects_to_valid_v1():
    raw = json.loads((Path(__file__).parent / "fixtures/tool-timing-canonical-failure.json").read_text())[
        "ng_trajectory"
    ]
    TrajectoryRecord.model_validate(raw)
    projected = project_trajectory_v1(raw)
    decoded = TrajectoryRecord.model_validate(projected)
    assert decoded.tool_calls[0].duration_ms == 10
    assert not {"span_id", "failure_id", "timings"} & projected["tool_calls"][0].keys()


@pytest.mark.parametrize("count", [0, 1, 2])
def test_projection_never_selects_flat_interval_among_multiple_boundaries(count):
    measurements = [timing().model_dump(), timing().model_copy(update={"boundary": "executor"}).model_dump()][:count]
    raw = {
        "tool_calls": [
            {
                "kind": "tool_call",
                "started_at": 1,
                "completed_at": 2,
                "duration_ms": 1000,
                "timing_source": "harness",
                "timings": measurements,
            }
        ]
    }
    projected = project_trajectory_v1(raw)["tool_calls"][0]
    assert projected["duration_ms"] == {0: 1000, 1: 2000, 2: None}[count]
    if count == 2:
        assert all(projected[key] is None for key in ("started_at", "completed_at", "duration_ms", "timing_source"))


@pytest.mark.parametrize("container", ["bundle", "trajectory"])
def test_duplicate_invocation_ids_across_instances_are_explicitly_unsupported(container):
    owners = [AgentInvocation(invocation_id="root", agent_instance=instance) for instance in ("a", "b")]
    with pytest.raises(ValidationError, match="invocation_id must be unique"):
        if container == "bundle":
            AgentObservationBundle(schema_version="2.0", source="test", records=owners)
        else:
            TrajectoryRecord(schema_version="2.0", task_id="task", capture_key="capture", invocations=owners)


def test_capture_join_keeps_each_adapter_span_diagnostic():
    from nemo_gym.trajectory.trajectory_record import join_agent_model_calls

    bundle = AgentObservationBundle(
        schema_version="2.0",
        source="test",
        records=[AgentInvocation(invocation_id="root")],
        gaps=[
            ObservationGap(code="tool_start_unavailable", invocation_id="root", span_id=span)
            for span in ("one", "two", "one")
        ],
    )
    for _ in range(3):
        bundle = join_agent_model_calls(bundle, [])
        assert {gap.span_id for gap in bundle.gaps} == {"one", "two"}
        assert len(bundle.gaps) == 2


@pytest.mark.parametrize("initial_instance", [None, "assistant"])
@pytest.mark.parametrize("owner_instance", [None, "assistant", "other"])
def test_output_join_preserves_unique_owner_with_partial_instance_evidence(initial_instance, owner_instance):
    invocation = AgentInvocation(
        invocation_id="root",
        agent_instance=owner_instance,
        conversation=[{"type": "function_call_output", "call_id": "normalized", "output": "usable result"}],
    )
    observed = tool(instance=initial_instance, model_call_ref=ModelCallRef(model_call_id="model"))
    tools, gaps = normalize([observed], invocations=[invocation])
    if owner_instance == "other" and initial_instance is not None:
        # A known contradictory owner cannot lend its output to this attempt.
        assert tools[0].output is None
        assert any(g.code == "tool_agent_instance_mismatch" for g in gaps)
    else:
        assert tools[0].output == "usable result"
        assert tools[0].agent_instance == (owner_instance or "assistant")
        assert tools[0].model_call_id == (None if owner_instance == "other" else "model")


@pytest.mark.parametrize("reverse", [False, True])
def test_snapshot_outcome_and_output_advance_together(reverse):
    from nemo_gym.trajectory.trajectory_record import TrajectoryToolCall

    snapshots = [
        TrajectoryToolCall(**tool(status="incomplete").model_dump(), output="started"),
        TrajectoryToolCall(**tool(status="completed").model_dump(), output="done"),
    ]
    if reverse:
        snapshots.reverse()
    retained, gaps = normalize(snapshots)
    assert (retained[0].span_id, retained[0].status, retained[0].output) == ("attempt", "completed", "done")
    assert "tool_output_ambiguous" not in {gap.code for gap in gaps}
    repeated, _ = normalize([*retained, *snapshots])
    assert len(repeated) == 1
    assert (repeated[0].span_id, repeated[0].status, repeated[0].output, repeated[0].timings) == (
        retained[0].span_id,
        retained[0].status,
        retained[0].output,
        retained[0].timings,
    )


@pytest.mark.parametrize("later_status", ["completed", "failed"])
def test_conflicting_terminal_snapshot_cannot_replace_output(later_status):
    from nemo_gym.trajectory.trajectory_record import TrajectoryToolCall

    retained, gaps = normalize(
        [
            TrajectoryToolCall(**tool(status="completed").model_dump(), output="done"),
            TrajectoryToolCall(**tool(status=later_status).model_dump(), output="different"),
        ]
    )
    assert (retained[0].status, retained[0].output) == ("completed", "done")
    expected = "tool_output_ambiguous" if later_status == "completed" else "tool_outcome_conflict"
    assert expected in {gap.code for gap in gaps}


@pytest.mark.parametrize("parents", [{"a": "a"}, {"a": "b", "b": "a"}, {"a": "b", "b": "c", "c": "a"}])
def test_parent_cycles_are_rejected_and_raw_edges_are_salvaged(parents):
    from nemo_gym.trajectory.trajectory_record import TrajectoryToolCall

    raw = [
        ToolCallObservation(invocation_id="root", tool_call_id=span, span_id=span, parent_span_id=parent)
        for span, parent in {**parents, "incoming": "a", "valid": None}.items()
    ]
    with pytest.raises(ValidationError, match="must not contain cycles"):
        TrajectoryRecord(
            schema_version="2.0",
            task_id="task",
            rollout_id="capture",
            tool_calls=[TrajectoryToolCall(**tool.model_dump()) for tool in raw],
        )
    retained, gaps = normalize_tool_calls(
        raw,
        capture_key="capture",
        producer_id="test",
        model_calls=[],
        invocations=[AgentInvocation(invocation_id="root")],
    )
    by_call = {tool.tool_call_id: tool for tool in retained}
    assert all(by_call[span].parent_span_id is None for span in parents)
    assert by_call["incoming"].parent_span_id == by_call["a"].span_id
    assert by_call["valid"].parent_span_id is None
    cycle_gaps = [gap for gap in gaps if gap.code == "tool_parent_cycle"]
    assert {gap.span_id for gap in cycle_gaps} == {by_call[span].span_id for span in parents}
    assert {gap.detail for gap in cycle_gaps} == {by_call[parent].span_id for parent in parents.values()}
    decoded = TrajectoryRecord(
        schema_version="2.0", task_id="task", rollout_id="capture", tool_calls=retained, gaps=gaps
    )
    assert TrajectoryRecord.model_validate(decoded.model_dump(mode="json")) == decoded
    repeated, _ = normalize_tool_calls(
        retained,
        capture_key="capture",
        producer_id="test",
        model_calls=[],
        invocations=[AgentInvocation(invocation_id="root")],
    )
    assert [(tool.span_id, tool.parent_span_id) for tool in repeated] == [
        (tool.span_id, tool.parent_span_id) for tool in retained
    ]


def test_completed_snapshot_removes_only_its_scoped_completion_gap():
    observed = tool(status="completed")
    retained, gaps = normalize(
        [observed],
        gaps=[
            ObservationGap(code="tool_completion_unavailable", invocation_id="root", span_id="attempt"),
            ObservationGap(code="tool_completion_unavailable", invocation_id="root"),
            ObservationGap(code="unrecognized_missing_evidence", invocation_id="root", span_id="attempt"),
        ],
    )
    assert [(gap.code, gap.span_id) for gap in gaps if gap.code == "tool_completion_unavailable"] == [
        ("tool_completion_unavailable", None)
    ]
    assert any(gap.code == "unrecognized_missing_evidence" and gap.span_id == retained[0].span_id for gap in gaps)
