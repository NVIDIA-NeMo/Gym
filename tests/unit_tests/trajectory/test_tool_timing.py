# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_gym.core.operation_timing import OperationTiming
from nemo_gym.rollouts.health.timeouts import read_rollout_timing
from nemo_gym.rollouts.rollout_collection import enrich_run_result
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
    records, _ = normalize(
        [tool(instance="a"), tool(instance="b")],
        invocations=[AgentInvocation(invocation_id="root", agent_instance=x) for x in ("a", "b")],
        calls=[call("a", instance="a"), call("b", instance="b")],
    )
    assert {(t.agent_instance, t.model_call_id) for t in records} == {("a", "a"), ("b", "b")}
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


def test_summary_selects_boundary_and_does_not_align_unknown_clocks():
    trajectory = {
        "schema_version": "2.0",
        "tool_calls": [{"timings": [timing().model_dump()]}, {"timings": [timing().model_dump()]}],
    }
    assert read_rollout_timing({"ng_trajectory": trajectory}).tool_s is None
    for item in trajectory["tool_calls"]:
        item["timings"][0]["clock_domain"] = "producer/one/epoch"
    assert read_rollout_timing({"ng_trajectory": trajectory}).tool_s == 2
    assert read_rollout_timing({"ng_trajectory": trajectory}, tool_boundary="executor").tool_s is None
    assert read_rollout_timing({"ng_trajectory": {"schema_version": "2.0", "tool_calls": []}}).tool_s is None


async def test_enrichment_preserves_join_before_omitting_payloads_and_is_replay_safe():
    invocation = AgentInvocation(invocation_id="root", agent_instance="assistant")
    bundle = AgentObservationBundle(
        schema_version="2.0", source="test", records=[invocation, tool(status="completed")]
    )
    raw = {
        "reward": 1,
        "response": {"output": []},
        "ng_agent_observations": bundle.model_dump(mode="json"),
        "ng_model_call_capture": {
            "calls": [
                {
                    "model_call_id": "model",
                    "agent_instance": "assistant",
                    "client_session_id": "root",
                    "client_assistant_message_id": "message",
                    "response": call().response,
                }
            ]
        },
    }
    row = {"task_id": "task", "_ng_rollout_id": "0-0-a1"}
    result = deepcopy(raw)
    await enrich_run_result(
        row,
        result,
        capture_dirs=[],
        observability_enabled=True,
        include_payloads=False,
        trajectory_schema_version="2.0",
    )
    saved = result["ng_trajectory"]
    assert saved["tool_calls"][0]["model_call_id"] == "model"
    assert saved["model_calls"][0]["response"] is None
    await enrich_run_result(
        row,
        result,
        capture_dirs=[],
        observability_enabled=True,
        include_payloads=False,
        trajectory_schema_version="2.0",
    )
    assert len(result["ng_trajectory"]["tool_calls"]) == 1
    assert result["reward"] == 1 and result["response"] == raw["response"]
    legacy = deepcopy(raw)
    await enrich_run_result(row, legacy, capture_dirs=[], observability_enabled=True)
    assert legacy["ng_trajectory"]["schema_version"] == "1.0"
    assert "span_id" not in legacy["ng_trajectory"]["tool_calls"][0]


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


@pytest.mark.asyncio
async def test_legacy_observation_upgrades_before_v2_serialization():
    row = {"task_id": "task", "_ng_rollout_id": "capture"}
    result = {
        "ng_agent_observations": {
            "source": "legacy",
            "records": [
                {"kind": "agent_invocation", "invocation_id": "root"},
                {
                    "kind": "tool_call",
                    "invocation_id": "root",
                    "tool_call_id": "request",
                    "started_at": 1.0,
                    "completed_at": 2.0,
                    "timing_source": "harness",
                },
            ],
        }
    }
    await enrich_run_result(row, result, capture_dirs=[], observability_enabled=True, trajectory_schema_version="2.0")
    decoded = TrajectoryRecord.model_validate(result["ng_trajectory"])
    assert decoded.tool_calls[0].span_id
    assert decoded.tool_calls[0].timings[0].started_at == 1.0


@pytest.mark.parametrize("fixture", ["legacy-artifact", "canonical-failure"])
async def test_saved_migration_evidence_survives_repeated_enrichment(fixture):
    result = json.loads((Path(__file__).parent / "fixtures" / f"tool-timing-{fixture}.json").read_text())
    row = {"task_id": "task", "_ng_rollout_id": "capture"}
    first_tools = None
    for _ in range(3):
        await enrich_run_result(
            row, result, capture_dirs=[], observability_enabled=True, trajectory_schema_version="2.0"
        )
        trajectory = TrajectoryRecord.model_validate(result["ng_trajectory"])
        assert len(trajectory.tool_calls) == 1
        tools = result["ng_trajectory"]["tool_calls"]
        if first_tools is not None:
            assert tools == first_tools
        first_tools = deepcopy(tools)
    if fixture == "canonical-failure":
        assert trajectory.tool_calls[0].failure_id == "failure"
        assert trajectory.failure_events["failure"]["failure_reason"] == "tool failed"
    else:
        assert not trajectory.tool_calls[0].timings
        assert any(
            g.code == "tool_legacy_boundary_unavailable" and '"timing_source": "artifact"' in g.detail
            for g in trajectory.gaps
        )


async def test_failure_event_merge_preserves_canonical_event_and_reports_conflict():
    result = json.loads((Path(__file__).parent / "fixtures/tool-timing-canonical-failure.json").read_text())
    result["failure_events"] = {
        "failure": {"boundary": "harness", "failure_kind": "example:failure", "failure_reason": "conflicting"},
        "other": {
            "boundary": "agent",
            "failure_kind": "example:other",
            "failure_reason": "wrapper",
            "cause_id": "failure",
        },
    }
    await enrich_run_result(
        {"task_id": "task", "_ng_rollout_id": "capture"},
        result,
        capture_dirs=[],
        observability_enabled=True,
        trajectory_schema_version="2.0",
    )
    trajectory = TrajectoryRecord.model_validate(result["ng_trajectory"])
    assert trajectory.failure_events["failure"]["failure_reason"] == "tool failed"
    assert trajectory.failure_events["other"]["cause_id"] == "failure"
    assert any(g.code == "failure_event_conflict" and g.detail == "failure" for g in trajectory.gaps)


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


@pytest.mark.parametrize(
    "response,status", [("upstream unavailable", 503), ("malformed", 200), ({"error": "failed"}, 200)]
)
@pytest.mark.parametrize("include_success", [False, True])
async def test_unusable_capture_cannot_remove_tool_evidence_or_obscure_success(response, status, include_success):
    result = {
        "ng_agent_observations": AgentObservationBundle(
            schema_version="2.0",
            source="test",
            records=[
                AgentInvocation(invocation_id="root", agent_instance="assistant"),
                tool(status="completed"),
            ],
        ).model_dump(mode="json"),
        "ng_model_call_capture": {
            "calls": [
                {
                    "model_call_id": "bad",
                    "client_session_id": "root",
                    "client_assistant_message_id": "message",
                    "status_code": status,
                    "response_raw": response,
                }
            ]
        },
    }
    if include_success:
        result["ng_model_call_capture"]["calls"].append(
            {
                "model_call_id": "good",
                "client_session_id": "root",
                "client_assistant_message_id": "message",
                "status_code": 200,
                "response": call().response,
            }
        )
    await enrich_run_result(
        {"task_id": "task", "_ng_rollout_id": "capture"},
        result,
        capture_dirs=[],
        observability_enabled=True,
        trajectory_schema_version="2.0",
    )
    decoded = TrajectoryRecord.model_validate(result["ng_trajectory"])
    (observed,) = decoded.tool_calls
    assert observed.timings[0].duration_ms == 2000
    assert observed.model_call_id == ("good" if include_success else None)
    assert not any(g.code == "trajectory_projection_failed" for g in decoded.gaps)
    assert any(g.code in {"tool_request_payload_malformed", "tool_request_payload_unsuccessful"} for g in decoded.gaps)


def assert_gap_targets(tools, gaps):
    for gap in gaps:
        if gap.span_id is not None:
            targets = [item for item in tools if gap.matches_tool(item)]
            assert len(targets) == 1
            assert (gap.agent_instance, gap.invocation_id) == (targets[0].agent_instance, targets[0].invocation_id)


@pytest.mark.parametrize("instances", [("a", "b"), (None, None)])
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


async def test_delayed_producer_does_not_rewrite_canonical_gaps_and_canonical_only_replay_is_stable():
    observed = tool("shared", instance=None).model_copy(update={"source_message_id": None})
    bundle = AgentObservationBundle(
        schema_version="2.0",
        source="test",
        records=[
            AgentInvocation(invocation_id="root"),
            observed,
        ],
        gaps=[ObservationGap(code="adapter_only", invocation_id="root", span_id="shared", detail="original")],
    )
    row = {"task_id": "task", "_ng_rollout_id": "capture"}
    result = {"ng_agent_observations": bundle.model_dump(mode="json")}

    async def enrich():
        await enrich_run_result(
            row, result, capture_dirs=[], observability_enabled=True, trajectory_schema_version="2.0"
        )
        return TrajectoryRecord.model_validate(result["ng_trajectory"])

    first = await enrich()
    (canonical_gap,) = [gap for gap in first.gaps if gap.code == "adapter_only"]
    assert canonical_gap.span_id == first.tool_calls[0].span_id
    bundle.records.append(observed.model_copy(update={"producer_id": "another"}))
    result["ng_agent_observations"] = bundle.model_dump(mode="json")
    second = await enrich()
    assert canonical_gap in second.gaps
    diagnostic = [gap for gap in second.gaps if gap.code == "adapter_only"]
    assert len(diagnostic) == 2
    assert sum(gap.span_id is not None for gap in diagnostic) == 1
    assert "ambiguous" in next(gap.detail for gap in diagnostic if gap.span_id is None)
    assert_gap_targets(second.tool_calls, second.gaps)
    serialized = json.loads(json.dumps(result["ng_trajectory"]))
    result.clear()
    result["ng_trajectory"] = serialized
    for _ in range(3):
        replay = await enrich()
        assert replay.tool_calls == second.tool_calls
        assert replay.gaps == second.gaps
        assert canonical_gap in replay.gaps


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
