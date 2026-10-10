# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The projection of a rollout row into spans: every span the analysis reads, on the right clock."""

from __future__ import annotations

from ng_latency.project import project_rollout

from tests.unit_tests.latency.conftest import NODE, SANDBOX, T0


def _by_name(timeline: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for span in timeline["spans"]:
        out.setdefault(span["name"], []).append(span)
    for spans in out.values():
        spans.sort(key=lambda s: s["start_wall_s"])
    return out


def test_terminus_row_projects_rollout_phases_agent_calls_attempts_batches_and_compaction(terminus_record):
    timeline = project_rollout(terminus_record)
    assert (timeline["task_id"], timeline["rollout_id"], timeline["agent_kind"]) == ("task-4", "4-1", "terminus2")
    spans = _by_name(timeline)
    [rollout] = spans["rollout"]
    assert rollout["clock"]["domain"] == NODE and rollout["end_kind"] == "derived"
    assert rollout["duration_ns"] == 94_999_500_000 and rollout["attrs"]["wall_latency_ms"] == 95_000.0
    assert {s["name"] for s in timeline["spans"]} >= {"seed_session", "agent_setup", "agent_execute", "verify"}
    [agent] = spans["agent"]
    assert (agent["start_wall_s"], agent["end_wall_s"]) == (T0 + 19.0, T0 + 72.0) and agent[
        "duration_ns"
    ] == 53_000_000_000
    assert agent["parent_span_id"] == rollout["span_id"] and agent["ids"]["invocation_id"] == "root"
    calls = spans["model_call"]
    assert [c["attrs"]["purpose"] for c in calls] == ["agent_step", "compaction_summary", "agent_step"]
    assert calls[0]["duration_ns"] == 10_000_000_000 and calls[0]["attrs"]["parsed_commands"] == 2
    assert calls[2]["attrs"]["task_complete"] is True and calls[2]["ids"]["client_call_id"] == "t2-3"
    attempts = [a for a in spans["model_call_attempt"] if a["parent_span_id"] == calls[2]["span_id"]]
    assert [a["status"] for a in attempts] == ["timeout", "ok"] and attempts[0]["attrs"]["error"] == "TimeoutError"
    servers = spans["model_call_server"]
    assert len(servers) == 3 and servers[0]["parent_span_id"] == calls[0]["span_id"]
    assert calls[0]["attrs"]["server_span_id"] == servers[0]["span_id"]
    assert servers[0]["attrs"]["tokens"] == {
        "input": 1000,
        "output": 40,
        "reasoning": 20,
        "completion_total": 60,
        "cached": 200,
    }
    vllm = servers[0]["attrs"]["vllm"]
    assert (vllm["queue_s"], vllm["prefill_s"], vllm["decode_s"], vllm["e2e_s"]) == (0.05, 1.0, 8.8, 9.85)
    assert abs(servers[0]["attrs"]["gym_hop_overhead_s"] - (9.98 - 9.85)) < 1e-9
    batches = spans["tool_batch"]
    assert len(batches) == 2 and batches[0]["attrs"]["num_commands"] == 2 and batches[0]["attrs"]["timeout"] is False
    assert (batches[0]["start_wall_s"], batches[0]["end_wall_s"]) == (T0 + 30.3, T0 + 32.4)
    assert batches[0]["attrs"]["response_received_wall_s"] == T0 + 32.5
    assert batches[0]["attrs"]["observation_trajectory_index"] == 3
    tools = spans["tool_call"]
    assert [t["parent_span_id"] for t in tools] == [
        batches[0]["span_id"],
        batches[0]["span_id"],
        batches[1]["span_id"],
    ]
    assert tools[0]["duration_ns"] == 1_000_000_000 and tools[0]["attrs"]["requested_wait_s"] == 1.0
    [compaction] = spans["compaction"]
    assert compaction["attrs"] == {
        "reason": "proactive",
        "outcome": "completed",
        "tokens_before": 1080,
        "tokens_after": 2000,
        "rejected_request_tokens": None,
    }
    assert compaction["duration_ns"] == 6_500_000_000
    assert timeline["clock_offsets"] == {} and timeline["flags"] == []


def test_opencode_row_projects_steps_subagent_tools_and_the_probe(opencode_record):
    timeline = project_rollout(opencode_record)
    spans = _by_name(timeline)
    [agent] = spans["agent"]
    assert agent["clock"]["domain"] == SANDBOX and agent["duration_ns"] is None
    # agent_end is the last assistant completion; the process marks stay beside it, outside the span
    assert agent["end_wall_s"] == T0 + 0.3 + 66.0 and agent["attrs"]["process_exit_wall_s"] == T0 + 0.3 + 70.0
    assert agent["attrs"]["opencode_startup_s"] == 12.0 and agent["attrs"]["opencode_shutdown_s"] == 4.0
    [sub] = spans["subagent"]
    assert sub["parent_span_id"] == agent["span_id"] and sub["ids"]["parent_invocation_id"] == "root"
    assert (
        sub["attrs"]["requested_wall_s"] == T0 + 0.3 + 33.2 and sub["attrs"]["spawned_by_tool_call_id"] == "call-task"
    )
    steps = spans["model_step"]
    assert len(steps) == 3 and steps[0]["attrs"]["first_part_wall_s"] == T0 + 0.3 + 33.0
    assert (steps[0]["start_wall_s"], steps[0]["end_wall_s"]) == (T0 + 0.3 + 25.0, T0 + 0.3 + 33.6)
    calls = spans["model_call"]
    assert calls[0]["parent_span_id"] == steps[0]["span_id"] and calls[0]["clock"]["domain"] == SANDBOX
    assert calls[0]["duration_ns"] is None and calls[0]["end_wall_s"] == T0 + 0.3 + 33.0  # first part arrival
    servers = spans["model_call_server"]
    assert servers[0]["clock"]["domain"] == NODE and servers[0]["duration_ns"] == 7_700_000_000
    tools = spans["tool_call"]
    assert [t["attrs"]["tool"] for t in tools] == ["bash", "task"]
    assert all(t["parent_span_id"] == steps[0]["span_id"] for t in tools)  # requested during step 1
    assert tools[0]["duration_ns"] is None and tools[0]["clock"]["domain"] == SANDBOX
    assert tools[0]["attrs"]["response_received_wall_s"] == T0 + 0.3 + 61.0
    assert timeline["clock_offsets"][SANDBOX]["offset_s"] == 0.3
    assert timeline["clock_offsets"][SANDBOX]["uncertainty_s"] == 0.04
    assert abs(timeline["clock_offsets"][SANDBOX]["offset_plus_min_latency_s"] - (-0.1)) < 1e-6  # server start − send
    assert timeline["clock_offsets"][SANDBOX]["samples"] == 3
    assert timeline["flags"] == ["tool_response_boundary_approximate"]


def test_a_killed_opencode_process_and_a_missing_probe_are_flagged(opencode_record):
    records = opencode_record["ng_agent_observations"]["records"]
    records[1] = {**records[1], "outcome": "timeout", "clock_offset_s": None, "clock_offset_uncertainty_s": None}
    opencode_record["ng_trajectory"]["invocations"][0]["process_exited_at"] = None
    timeline = project_rollout(opencode_record)
    [agent] = _by_name(timeline)["agent"]
    assert agent["status"] == "timeout" and agent["attrs"]["opencode_shutdown_s"] is None
    assert set(timeline["flags"]) >= {"agent_end_unobserved", "clock_offset_unavailable"}
    assert "offset_s" not in timeline["clock_offsets"][SANDBOX]


def test_rows_without_observability_records_project_to_nothing():
    assert project_rollout({"_ng_task_index": 1, "_ng_rollout_index": 0, "reward": 0.0}) is None


def test_a_terminus_agent_stopped_by_the_sandbox_cap_is_a_timeout(terminus_record):
    """Harbor's loop is cancelled at the sandbox cap: the invocation ends incomplete with a TimeoutError. The agent
    span says timeout, like an OpenCode process killed at the same cap, so the report counts both alike."""
    terminus_record["ng_trajectory"]["invocations"][0].update(status="incomplete", error_type="TimeoutError")
    agent = next(s for s in project_rollout(terminus_record)["spans"] if s["name"] == "agent")
    assert agent["status"] == "timeout" and agent["attrs"]["error_type"] == "TimeoutError"


def test_compaction_model_calls_are_children_of_their_compaction(terminus_record, opencode_record):
    """The document asks for compaction as a parent span with its child work beneath it. Terminus names the calls it
    made during a compaction; OpenCode does not, so a summary step inside the compaction's window is its child."""
    spans = project_rollout(terminus_record)["spans"]
    [compaction] = [s for s in spans if s["name"] == "compaction"]
    summary = next(s for s in spans if s["name"] == "model_call" and s["attrs"]["purpose"] == "compaction_summary")
    assert summary["parent_span_id"] == compaction["span_id"]
    assert all(
        s["parent_span_id"] != compaction["span_id"] for s in spans if s["name"] == "model_call" and s is not summary
    )

    calls = opencode_record["ng_trajectory"]["model_calls"]
    calls[2]["model_call_purpose"] = "compaction_summary"
    opencode_record["ng_trajectory"]["compactions"].append(
        {
            "invocation_id": "root",
            "observed_at": calls[2]["started_at"] - 0.5,
            "completed_at": calls[2]["completed_at"] + 0.5,
            "trigger": "overflow",
            "outcome": "completed",
            "clock_domain": calls[2]["clock_domain"],
        }
    )
    spans = project_rollout(opencode_record)["spans"]
    [compaction] = [s for s in spans if s["name"] == "compaction"]
    by_id = {s["span_id"]: s for s in spans}
    summary = next(s for s in spans if s["name"] == "model_call" and s["ids"]["model_call_id"] == "cap-3")
    step = by_id[summary["parent_span_id"]]
    assert step["name"] == "model_step" and step["parent_span_id"] == compaction["span_id"]
    other = next(s for s in spans if s["name"] == "model_call" and s["ids"]["model_call_id"] == "cap-1")
    assert by_id[other["parent_span_id"]]["parent_span_id"] != compaction["span_id"]


def test_an_opencode_step_the_killed_process_never_finished_is_incomplete(opencode_record):
    """The request was in flight when the sandbox cap killed the process: the step has a send time and no end. The
    harness never saw it finish, so the call is incomplete, not ok; the server's own view (cancelled) stands beside."""
    records = opencode_record["ng_agent_observations"]["records"]
    records[1] = {**records[1], "outcome": "timeout"}
    opencode_record["ng_trajectory"]["invocations"][0]["process_exited_at"] = None
    calls = opencode_record["ng_trajectory"]["model_calls"]
    last = calls[2]
    calls.append(
        {
            **last,
            "model_call_id": "cap-4",
            "started_at": last["completed_at"] + 2.0,
            "completed_at": None,
            "duration_ms": None,
            "processed_at": None,
            "server_started_at": last["server_completed_at"] + 2.1,
            "server_completed_at": last["server_completed_at"] + 6.0,
            "server_duration_ms": 3_900.0,
            "status": None,
            "response_metadata": {**last["response_metadata"], "response_id": "resp-4", "error_category": "cancelled"},
        }
    )
    spans = project_rollout(opencode_record)["spans"]
    call = next(s for s in spans if s["name"] == "model_call" and s["ids"]["model_call_id"] == "cap-4")
    server = next(s for s in spans if s["name"] == "model_call_server" and s["ids"]["model_call_id"] == "cap-4")
    assert (call["status"], call["end_wall_s"], server["status"]) == ("incomplete", None, "cancelled")
