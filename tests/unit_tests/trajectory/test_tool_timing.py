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


@pytest.mark.parametrize("domain", [[], {}, ""])
@pytest.mark.parametrize("kind", ["tool", "model"])
def test_malformed_clock_domains_leave_union_unknown(domain, kind):
    record = {"ng_trajectory": {"schema_version": "2.0", "tool_calls": []}}
    if kind == "tool":
        record["ng_trajectory"]["tool_calls"] = [
            {"timings": [{"boundary": "lifecycle", "started_at": 1, "completed_at": 2, "clock_domain": d}]}
            for d in (domain, "clock")
        ]
    else:
        record["ng_model_call_capture"] = {
            "calls": [{"started_at": 1, "completed_at": 2, "clock_domain": d} for d in (domain, "clock")]
        }
    result = read_rollout_timing(record)
    assert (result.tool_s if kind == "tool" else result.policy_s) is None


@pytest.mark.parametrize("container", ["bundle", "trajectory"])
def test_duplicate_invocation_ids_across_instances_are_explicitly_unsupported(container):
    owners = [AgentInvocation(invocation_id="root", agent_instance=instance) for instance in ("a", "b")]
    with pytest.raises(ValidationError, match="invocation_id must be unique"):
        if container == "bundle":
            AgentObservationBundle(schema_version="2.0", source="test", records=owners)
        else:
            TrajectoryRecord(schema_version="2.0", task_id="task", capture_key="capture", invocations=owners)


@pytest.mark.parametrize("include_payloads", [False, True])
async def test_projection_failure_respects_payload_policy_without_losing_other_evidence(include_payloads):
    result = {
        "reward": 0.25,
        "response": {"output": "primary"},
        "ng_model_call_capture": {
            "calls": [
                {
                    "model_call_id": "model",
                    "latency_total_ms": -1,
                    "request": {"input": "question"},
                    "response": {"output": "answer"},
                    "request_raw": "raw request",
                    "response_raw": "raw response",
                }
            ]
        },
    }
    await enrich_run_result(
        {"task_id": "task", "_ng_rollout_id": "capture"},
        result,
        capture_dirs=[],
        observability_enabled=True,
        trajectory_schema_version="2.0",
        include_payloads=include_payloads,
    )
    captured = result["ng_model_call_capture"]["calls"][0]
    assert any(gap["code"] == "trajectory_projection_failed" for gap in result["ng_model_call_capture"]["gaps"])
    assert bool({"request", "response", "request_raw", "response_raw"} & captured.keys()) == include_payloads
    assert result["reward"] == 0.25 and result["response"] == {"output": "primary"}


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


async def test_shared_failure_report_fields_link_and_survive_replay():
    from nemo_gym.core.failure_records import FailureReport
    from nemo_gym.core.failure_transport import FailureReportRow

    events = {
        "failure": {
            "boundary": "resources_server",
            "failure_kind": "example:failure",
            "failure_reason": "tool failed",
            "exception_type": "ResourceError",
            "operation": "execute",
            "operation_id": "operation",
            "attempt_id": "attempt",
        }
    }
    report = FailureReport(events=events, failure_id="failure")
    bundle = AgentObservationBundle(
        schema_version="2.0",
        source="test",
        records=[
            AgentInvocation(invocation_id="root", agent_instance="assistant"),
            tool(status="failed", failure_id="failure"),
        ],
    )
    result = {"ng_agent_observations": bundle.model_dump(mode="json"), **FailureReportRow.fields(report)}
    for _ in range(3):
        await enrich_run_result(
            {"task_id": "task", "_ng_rollout_id": "capture"},
            result,
            capture_dirs=[],
            observability_enabled=True,
            trajectory_schema_version="2.0",
        )
        record = TrajectoryRecord.model_validate(result["ng_trajectory"])
        assert record.tool_calls[0].failure_id == "failure"
        assert record.failure_events["failure"]["operation_id"] == "operation"
        assert not any(g.code == "failure_event_conflict" for g in record.gaps)


async def test_default_enrichment_projects_valid_v2_failure_evidence_to_v1():
    result = json.loads((Path(__file__).parent / "fixtures/tool-timing-canonical-failure.json").read_text())
    await enrich_run_result(
        {"task_id": "task", "_ng_rollout_id": "capture"}, result, capture_dirs=[], observability_enabled=True
    )
    record = TrajectoryRecord.model_validate(result["ng_trajectory"])
    assert record.schema_version == "1.0"
    assert record.tool_calls[0].duration_ms == 10
    assert not any(g.code == "trajectory_projection_failed" for g in record.gaps)


async def test_default_v1_keeps_unqualified_legacy_flat_timing():
    result = {
        "ng_agent_observations": {
            "source": "legacy",
            "records": [
                {
                    "kind": "tool_call",
                    "invocation_id": "root",
                    "tool_call_id": "request",
                    "started_at": 1.0,
                    "completed_at": 2.0,
                    "duration_ms": 1000.0,
                }
            ],
        }
    }
    await enrich_run_result(
        {"task_id": "task", "_ng_rollout_id": "capture"}, result, capture_dirs=[], observability_enabled=True
    )
    record = TrajectoryRecord.model_validate(result["ng_trajectory"])
    assert record.schema_version == "1.0" and record.tool_calls[0].duration_ms == 1000
    assert read_rollout_timing(result).tool_s == 1.0
    upgraded = deepcopy(result)
    await enrich_run_result(
        {"task_id": "task", "_ng_rollout_id": "capture"},
        upgraded,
        capture_dirs=[],
        observability_enabled=True,
        trajectory_schema_version="2.0",
    )
    assert read_rollout_timing(upgraded).tool_s is None
    assert "tool_legacy_boundary_unavailable" in {gap["code"] for gap in upgraded["ng_trajectory"]["gaps"]}


async def test_defaulted_legacy_observation_does_not_erase_completed_producer():
    from nemo_gym.rollouts.trajectory_builder import _build_trajectory_record

    observation = AgentObservationBundle(
        source="test", records=[ToolCallObservation(invocation_id="root", tool_call_id="request")]
    )
    result = {
        "ng_trajectory": {
            "schema_version": "1.0",
            "task_id": "task",
            "rollout_id": "capture",
            "invocations": [{"invocation_id": "root"}],
            "tool_calls": [
                {
                    "invocation_id": "root",
                    "tool_call_id": "request",
                    "status": "completed",
                    "started_at": 1,
                    "completed_at": 2,
                    "duration_ms": 1000,
                    "timing_source": "executor",
                }
            ],
        },
        "ng_agent_observations": observation.model_dump(mode="json"),
    }
    row = {"task_id": "task", "_ng_rollout_id": "capture"}
    for _ in range(3):
        result["ng_trajectory"] = _build_trajectory_record(row, result).model_dump(mode="json")
        saved = result["ng_trajectory"]["tool_calls"][0]
        assert (
            saved["status"],
            saved["started_at"],
            saved["completed_at"],
            saved["duration_ms"],
            saved["timing_source"],
        ) == ("completed", 1, 2, 1000, "executor")


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


@pytest.mark.asyncio
@pytest.mark.parametrize("schema_version", ["1.0", "2.0"])
async def test_reenrichment_advances_tool_and_conversation_from_same_snapshot(schema_version):
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    request = {"type": "function_call", "call_id": "normalized", "name": "bash", "arguments": "{}"}

    def bundle(status, output):
        invocation = AgentInvocation(
            invocation_id="root",
            agent_instance="assistant",
            conversation=[request, {"type": "function_call_output", "call_id": "normalized", "output": output}],
        )
        observed = AgentObservationBundle(
            schema_version="2.0", source="test", records=[invocation, tool(status=status)]
        ).model_dump(mode="json")
        if schema_version == "1.0":
            from nemo_gym.trajectory.trajectory_format import project_bundle_v1

            return project_bundle_v1(observed)
        return observed

    result = {"ng_agent_observations": bundle("incomplete", "started")}
    await enrich_run_result(
        row, result, capture_dirs=[], observability_enabled=False, trajectory_schema_version=schema_version
    )
    original_span = result["ng_trajectory"]["tool_calls"][0].get("span_id")
    for status, output in [("completed", "done"), ("incomplete", "started"), ("completed", "done")]:
        result["ng_agent_observations"] = bundle(status, output)
        await enrich_run_result(
            row, result, capture_dirs=[], observability_enabled=False, trajectory_schema_version=schema_version
        )
        retained = result["ng_trajectory"]["tool_calls"]
        assert len(retained) == 1
        assert (retained[0].get("span_id"), retained[0]["status"], retained[0]["output"]) == (
            original_span,
            "completed",
            "done",
        )
        conversation = result["ng_trajectory"]["invocations"][0]["conversation"]
        assert conversation[0]["arguments"] == "{}"
        assert conversation[1]["output"] == "done"
        if schema_version == "2.0":
            assert not any(
                gap["code"] == "tool_completion_unavailable" and gap.get("span_id") == original_span
                for gap in result["ng_trajectory"]["gaps"]
            )


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


@pytest.mark.parametrize("include_payloads", [False, True])
async def test_capture_generated_turns_respect_payload_omission(include_payloads):
    invocation = AgentInvocation(invocation_id="root", model_calls=[ModelCallRef(model_call_id="model")])
    result = {
        "response": {"output": "producer answer"},
        "reward": 1,
        "ng_agent_observations": AgentObservationBundle(
            schema_version="2.0", source="test", records=[invocation]
        ).model_dump(mode="json"),
        "ng_model_call_capture": {
            "calls": [
                {
                    "model_call_id": "model",
                    "request": {"input": "private prompt"},
                    "response": {
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "private answer"}],
                            },
                            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "private reasoning"}]},
                        ]
                    },
                }
            ]
        },
    }
    await enrich_run_result(
        {"task_id": "task", "_ng_rollout_id": "capture"},
        result,
        capture_dirs=[],
        observability_enabled=True,
        include_payloads=include_payloads,
        trajectory_schema_version="2.0",
    )
    turns = result["ng_trajectory"]["turns"]
    assert len(turns) == 1
    for value in ("private prompt", "private answer", "private reasoning"):
        assert (value in json.dumps(turns)) == include_payloads
    assert turns[0]["model_calls"][0]["model_call_id"] == "model"
    assert result["response"] == {"output": "producer answer"} and result["reward"] == 1


@pytest.mark.parametrize(
    "schema_version,retained_instance", [("1.0", None), ("2.0", None), ("2.0", "assistant"), ("2.0", "other")]
)
async def test_invocation_instance_enrichment_retains_one_owner(retained_instance, schema_version):
    result = {
        "ng_trajectory": {
            "schema_version": schema_version,
            "task_id": "task",
            "rollout_id": "capture",
            "invocations": [{"invocation_id": "root", "agent_instance": retained_instance, "status": "completed"}],
        },
        "ng_agent_observations": AgentObservationBundle(
            schema_version="2.0",
            source="test",
            records=[AgentInvocation(invocation_id="root", agent_instance="assistant")],
        ).model_dump(mode="json"),
    }
    await enrich_run_result(
        {"task_id": "task", "_ng_rollout_id": "capture"},
        result,
        capture_dirs=[],
        observability_enabled=True,
        trajectory_schema_version="2.0",
    )
    trajectory = TrajectoryRecord.model_validate(result["ng_trajectory"])
    assert len(trajectory.invocations) == 1
    assert trajectory.invocations[0].agent_instance == (retained_instance or "assistant")
    assert trajectory.invocations[0].status == "completed"
    assert ("invocation_agent_instance_conflict" in {gap.code for gap in trajectory.gaps}) == (
        retained_instance == "other"
    )


async def test_equivalent_failure_encodings_preserve_imported_causal_link():
    result = json.loads((Path(__file__).parent / "fixtures/tool-timing-canonical-failure.json").read_text())
    event = result["ng_trajectory"]["failure_events"]["failure"]
    result["failure_events"] = {"failure": {**event, "event_id": "failure", "cause_id": None}}
    result["ng_agent_observations"] = AgentObservationBundle(
        schema_version="2.0",
        source="test",
        records=[tool("imported", status="failed", failure_id="failure", instance=None)],
    ).model_dump(mode="json")
    await enrich_run_result(
        {"task_id": "task", "_ng_rollout_id": "capture"},
        result,
        capture_dirs=[],
        observability_enabled=True,
        trajectory_schema_version="2.0",
    )
    trajectory = TrajectoryRecord.model_validate(result["ng_trajectory"])
    assert len(trajectory.tool_calls) == 2
    assert all(tool.failure_id == "failure" for tool in trajectory.tool_calls)
    assert not any(gap.code == "failure_event_conflict" for gap in trajectory.gaps)


@pytest.mark.parametrize("include_payloads", [False, True])
async def test_payload_omission_preserves_independent_producer_turns(include_payloads):
    result = {
        "ng_trajectory": {
            "schema_version": "2.0",
            "task_id": "task",
            "capture_key": "capture",
            "invocations": [
                {
                    "invocation_id": "root",
                    "conversation": [{"type": "message", "role": "user", "content": "producer conversation"}],
                }
            ],
            "turns": [
                {
                    "invocation_id": "root",
                    "task_id": "task",
                    "rollout_id": "capture",
                    "turn_no": 1,
                    "timestamp": 1,
                    "step_count": 0,
                    "question": "producer question",
                    "answer": "producer answer",
                    "reasoning_content": "producer reasoning",
                }
            ],
        }
    }
    expected = TrajectoryRecord.model_validate(result["ng_trajectory"]).model_dump(mode="json")
    await enrich_run_result(
        {"task_id": "task", "_ng_rollout_id": "capture"},
        result,
        capture_dirs=[],
        observability_enabled=True,
        include_payloads=include_payloads,
        trajectory_schema_version="2.0",
    )
    saved = result["ng_trajectory"]
    for key in ("question", "answer", "reasoning_content"):
        assert saved["turns"][0][key] == expected["turns"][0][key]
    assert saved["invocations"][0]["conversation"] == expected["invocations"][0]["conversation"]


async def test_enrichment_keeps_missing_end_gap_until_timing_arrives():
    observed = tool(status="completed")
    observed.timings[0].completed_at = None
    observed.timings[0].duration_source = None
    observed.timings[0].duration_ms = None
    bundle = AgentObservationBundle(
        schema_version="2.0",
        source="test",
        records=[AgentInvocation(invocation_id="root", agent_instance="assistant"), observed],
        gaps=[ObservationGap(code="tool_completion_unavailable", invocation_id="root", span_id="attempt")],
    )
    result = {"ng_agent_observations": bundle.model_dump(mode="json")}
    row = {"task_id": "task", "_ng_rollout_id": "capture"}
    for _ in range(2):
        await enrich_run_result(
            row, result, capture_dirs=[], observability_enabled=True, trajectory_schema_version="2.0"
        )
        assert any(gap["code"] == "tool_completion_unavailable" for gap in result["ng_trajectory"]["gaps"])
    observed.timings = [timing()]
    result["ng_agent_observations"] = bundle.model_dump(mode="json")
    await enrich_run_result(row, result, capture_dirs=[], observability_enabled=True, trajectory_schema_version="2.0")
    assert not any(gap["code"] == "tool_completion_unavailable" for gap in result["ng_trajectory"]["gaps"])


async def test_conflicting_invocation_declines_its_tools_without_losing_other_evidence():
    result = {
        "ng_trajectory": {
            "schema_version": "2.0",
            "task_id": "task",
            "capture_key": "capture",
            "invocations": [{"invocation_id": "root", "agent_instance": "retained"}],
        },
        "ng_agent_observations": AgentObservationBundle(
            schema_version="2.0",
            source="test",
            records=[
                AgentInvocation(invocation_id="root", agent_instance="assistant"),
                tool(),
                AgentInvocation(invocation_id="unrelated", agent_instance="assistant"),
                tool("unrelated").model_copy(update={"invocation_id": "unrelated"}),
            ],
        ).model_dump(mode="json"),
    }
    await enrich_run_result(
        {"task_id": "task", "_ng_rollout_id": "capture"},
        result,
        capture_dirs=[],
        observability_enabled=True,
        trajectory_schema_version="2.0",
    )
    trajectory = TrajectoryRecord.model_validate(result["ng_trajectory"])
    assert len(trajectory.invocations) == 2
    assert [tool.invocation_id for tool in trajectory.tool_calls] == ["unrelated"]
    assert any(gap.code == "invocation_agent_instance_conflict" for gap in trajectory.gaps)
    assert len(result["ng_agent_observations"]["records"]) == 4
