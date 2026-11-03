# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from jsonschema import Draft202012Validator

from nemo_gym.trajectory.tool_events import read_tool_events, replay_tool_events, tool_event_json_schema
from nemo_gym.trajectory.tool_timing import normalize_tool_calls
from nemo_gym.trajectory.trajectory_record import TrajectoryModelCall
from nemo_gym.utils.tool_recording import ClockContext, ToolCallRecorder


def event(kind, *, producer="producer-1", span="span-1", sequence=None, **overrides):
    return {
        "schema_version": "2.0",
        "kind": kind,
        "producer_id": producer,
        "span_id": span,
        "sequence": sequence if sequence is not None else (0 if kind == "start" else 1),
        "invocation_id": "invocation-1",
        "agent_instance": "assistant",
        "native_call_id": "call-1",
        "tool_name": "bash",
        "boundary": "executor",
        "timing_source": "test.subprocess",
        "clock_domain": "sandbox/one/epoch-1",
        "model_call_ref": {"model_call_id": "model-1"},
        **(
            {"started_at": 100.0, "status": "incomplete"}
            if kind == "start"
            else {"completed_at": 102.0, "duration_ms": 1900.0, "duration_source": "monotonic", "status": "completed"}
        ),
        **overrides,
    }


def codes(gaps):
    return {gap.code for gap in gaps}


def test_delayed_reordered_and_duplicated_events_keep_producer_measurements():
    rows = [event("finish"), event("start"), event("finish")]
    tools, gaps = replay_tool_events(rows)
    assert not gaps
    assert len(tools) == 1
    assert tools[0].span_id == "span-1"
    measurement = tools[0].timings[0]
    assert (measurement.started_at, measurement.completed_at, measurement.duration_ms) == (100, 102, 1900)
    assert tools[0].model_call_ref.model_call_id == "model-1"


def test_conflicting_sequence_does_not_pick_first_copy():
    first = event("finish")
    conflicting = event("finish", completed_at=103.0)
    for rows in ([event("start"), first, conflicting], [conflicting, event("start"), first]):
        tools, gaps = replay_tool_events(rows)
        assert tools[0].status == "incomplete"
        assert tools[0].timings[0].completed_at is None
        assert codes(gaps) == {"tool_capture_sequence_conflict", "tool_completion_unavailable"}


def test_truncated_tail_preserves_flushed_start(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(event("start")) + '\n{"schema_version":')
    tools, gaps = read_tool_events(path)
    assert len(tools) == 1
    assert tools[0].status == "incomplete"
    assert tools[0].timings[0].started_at == 100
    assert {"tool_capture_record_invalid", "tool_completion_unavailable"}.issubset(codes(gaps))


def test_failed_outcome_survives_invalid_clock_and_duration():
    tools, gaps = replay_tool_events([event("start"), event("finish", completed_at=float("nan"), status="failed")])
    assert tools[0].status == "failed"
    assert tools[0].timings[0].completed_at is None
    assert tools[0].timings[0].duration_ms == 1900
    assert "tool_timing_invalid" in codes(gaps)


def test_new_clock_epoch_keeps_monotonic_duration_without_splicing_endpoints():
    tools, gaps = replay_tool_events([event("start"), event("finish", clock_domain="sandbox/one/epoch-2")])
    assert tools[0].timings[0].started_at == 100
    assert tools[0].timings[0].completed_at is None
    assert tools[0].timings[0].duration_ms == 1900
    assert "tool_clock_discontinuity" in codes(gaps)


def test_finish_without_start_and_duration_only_evidence_remain_useful():
    tools, gaps = replay_tool_events([event("finish", completed_at=None, duration_ms=0)])
    assert tools[0].status == "completed"
    assert tools[0].timings[0].started_at is None
    assert tools[0].timings[0].duration_ms == 0
    assert "tool_start_unavailable" in codes(gaps)


def test_same_native_id_distinct_attempts_and_producers_are_retained():
    rows = [
        event("start"),
        event("start", span="span-2", sequence=2),
        event("start", producer="producer-2", agent_instance=None),
    ]
    tools, gaps = replay_tool_events(rows)
    assert len(tools) == 3
    assert len({(tool.producer_id, tool.span_id) for tool in tools}) == 3
    assert all(tool.status == "incomplete" for tool in tools)
    assert "tool_completion_unavailable" in codes(gaps)


def test_event_replay_feeds_the_same_model_join_as_artifact_normalization():
    tools, _ = replay_tool_events([event("start"), event("finish")])
    call = TrajectoryModelCall(
        model_call_id="model-1",
        agent_instance="assistant",
        response={"output": [{"type": "function_call", "call_id": "call-1", "name": "bash", "arguments": "{}"}]},
    )
    normalized, gaps = normalize_tool_calls(
        tools, capture_key="task-0-a1", producer_id="collector", model_calls=[call], invocations=[]
    )
    assert normalized[0].model_call_id == "model-1"
    assert not any(gap.code.startswith("tool_model_call") for gap in gaps)
    replayed, _ = normalize_tool_calls(
        [*tools, *tools], capture_key="task-0-a1", producer_id="collector", model_calls=[call], invocations=[]
    )
    assert len(replayed) == 1
    assert replayed[0].span_id == normalized[0].span_id


def test_producer_records_validate_against_generated_schema(tmp_path):
    path = tmp_path / "events.jsonl"
    recorder = ToolCallRecorder(
        path=path,
        producer_id="p",
        invocation_id="i",
        agent_instance=None,
        clock=ClockContext(location="producer", owner="p"),
        boundary="lifecycle",
        timing_source="test.hooks",
    )
    with recorder.call(native_call_id="call", tool_name="bash") as handle:
        handle.set_outcome("failed", failure_id="f")
    validator = Draft202012Validator(tool_event_json_schema())
    for row in map(json.loads, path.read_text().splitlines()):
        validator.validate(row)
    tools, gaps = read_tool_events(path)
    assert not gaps
    assert tools[0].failure_id == "f"


@pytest.mark.parametrize("change", [{"schema_version": "3.0"}, {"status": "timed_out"}, {"sequence": True}])
def test_invalid_wire_contract_values_are_reported(change):
    tools, gaps = replay_tool_events([event("finish", **change)])
    assert tools == []
    assert "tool_capture_record_invalid" in codes(gaps)


@pytest.mark.parametrize("domain", ["sandbox/one/epoch-2", None])
@pytest.mark.parametrize("source", ["wall_clock", "monotonic", "reported"])
def test_incomparable_clocks_only_keep_independent_durations(domain, source):
    tools, gaps = replay_tool_events([event("start"), event("finish", clock_domain=domain, duration_source=source)])
    timing = tools[0].timings[0]
    assert timing.completed_at is None
    assert timing.duration_ms == (None if source == "wall_clock" else 1900)
    assert timing.duration_source == (None if source == "wall_clock" else source)
    assert codes(gaps) & {"tool_clock_discontinuity", "tool_clock_alignment_unavailable"}


@pytest.mark.parametrize(
    "metadata", [{"clock_domain": ""}, {"clock_domain": []}, {"duration_source": "bad"}, {"duration_source": {}}]
)
def test_invalid_optional_provenance_preserves_outcome_and_duration(metadata):
    tools, gaps = replay_tool_events(
        [event("start"), event("finish", status="failed", failure_id="failure-1", **metadata)]
    )
    assert tools[0].status == "failed"
    assert tools[0].failure_id == "failure-1"
    assert tools[0].timings[0].duration_ms == 1900
    assert any(gap.code == "tool_timing_invalid" and gap.span_id == "span-1" for gap in gaps)
    assert "tool_completion_unavailable" not in codes(gaps)


def test_producer_collision_survives_common_normalization_with_parent_links():
    rows = []
    for producer in ("one", "two"):
        rows.extend(
            [
                event("start", producer=producer, agent_instance=None),
                event(
                    "start", producer=producer, agent_instance=None, span="child", sequence=2, parent_span_id="span-1"
                ),
            ]
        )
    observations, _ = replay_tool_events(rows)
    tools, gaps = normalize_tool_calls(
        observations, capture_key="task-0-a1", producer_id="collector", model_calls=[], invocations=[]
    )
    assert len(tools) == 4
    assert len({tool.span_id for tool in tools}) == 4
    for producer in ("one", "two"):
        parent, child = [tool for tool in tools if tool.producer_id == producer]
        assert child.parent_span_id == parent.span_id
    assert "tool_attempt_identity_conflict" not in codes(gaps)


async def test_replay_collector_and_direct_enrichment_agree_across_instances_and_retries():
    from copy import deepcopy

    from nemo_gym.rollouts.rollout_collection import _build_trajectory_record, enrich_run_result
    from nemo_gym.trajectory.trajectory_record import AgentInvocation, AgentObservationBundle

    rows, calls, invocations = [], [], []
    for instance in ("assistant-a", "assistant-b"):
        invocations.append(AgentInvocation(invocation_id=instance, agent_instance=instance))
        for attempt in ("retry", "success"):
            calls.append(
                {
                    "model_call_id": f"{instance}-{attempt}",
                    "agent_instance": instance,
                    "response": {
                        "output": []
                        if attempt == "retry"
                        else [
                            {"type": "function_call", "call_id": native, "name": "bash", "arguments": "{}"}
                            for native in ("call-1", "call-2")
                        ]
                    },
                }
            )
        for index, native in enumerate(("call-1", "call-2")):
            for kind in ("start", "finish"):
                rows.append(
                    event(
                        kind,
                        producer=instance,
                        span=f"span-{index}",
                        sequence=2 * index + (kind == "finish"),
                        invocation_id=instance,
                        agent_instance=instance,
                        native_call_id=native,
                        model_call_ref={"model_call_id": f"{instance}-success"},
                    )
                )
    observations, gaps = replay_tool_events([*reversed(rows), *rows])
    bundle = AgentObservationBundle(
        schema_version="2.0", source="test", records=[*invocations, *observations], gaps=gaps
    )
    raw = {
        "reward": 1,
        "response": {"output": []},
        "ng_agent_observations": bundle.model_dump(mode="json"),
        "ng_model_call_capture": {"calls": calls},
    }
    row = {"task_id": "task", "_ng_rollout_id": "0-0-a1"}
    collector = _build_trajectory_record(row, deepcopy(raw), schema_version="2.0").model_dump(mode="json")
    direct = deepcopy(raw)
    await enrich_run_result(row, direct, capture_dirs=[], observability_enabled=True, trajectory_schema_version="2.0")
    assert direct["ng_trajectory"]["tool_calls"] == collector["tool_calls"]
    assert len(collector["tool_calls"]) == 4
    assert {tool["model_call_id"] for tool in collector["tool_calls"]} == {
        "assistant-a-success",
        "assistant-b-success",
    }
    before = deepcopy(direct["ng_trajectory"]["tool_calls"])
    await enrich_run_result(row, direct, capture_dirs=[], observability_enabled=True, trajectory_schema_version="2.0")
    assert direct["ng_trajectory"]["tool_calls"] == before
    assert direct["reward"] == raw["reward"] and direct["response"] == raw["response"]
