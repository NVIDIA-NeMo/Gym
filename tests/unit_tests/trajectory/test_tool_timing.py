# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

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


def normalize(tools, *, calls=None, invocations=None):
    return normalize_tool_calls(
        tools,
        capture_key="0-0-a1",
        producer_id="test",
        model_calls=calls or [call()],
        invocations=invocations or [AgentInvocation(invocation_id="root", agent_instance="assistant")],
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
    assert [(t.span_id, t.status, t.model_call_id) for t in records] == [
        ("first", "failed", "model"),
        ("second", "completed", "model"),
    ]
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
