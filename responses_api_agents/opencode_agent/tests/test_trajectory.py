# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3
from functools import partial

import pytest

from nemo_gym.config import ModelServerRef
from nemo_gym.rollouts import read_rollout_timing, run_health_checks
from nemo_gym.rollouts.trajectory_builder import _build_trajectory_record
from nemo_gym.trajectory import AgentInvocation, ModelCallRecord, TrajectoryRecord, join_agent_model_calls
from responses_api_agents.opencode_agent.app import _parse_opencode_session
from responses_api_agents.opencode_agent.tests.test_app import _session_db


@pytest.fixture(
    params=[
        partial(_parse_opencode_session, observation_schema_version="2.0"),
        partial(_parse_opencode_session, observation_schema_version="2.0", require_terminal_finish=True),
    ],
    ids=["local", "sandbox"],
)
def parse(request):
    return request.param


def _policy(*parts, **message):
    return (
        {"role": "assistant", "time": {"created": 1000, "completed": 2000}, **message},
        [{"type": "step-start"}, *parts, {"type": "step-finish"}],
    )


def _trajectory(parse, db):
    trajectory = TrajectoryRecord(task_id="0", rollout_id="0-0")
    observations = parse(db, "fallback", trajectory)
    return trajectory, observations


def _health(tmp_path, trajectory, observations, **call_updates):
    call = ModelCallRecord.model_validate(
        dict(
            call_index=0,
            model_call_id="captured",
            client_session_id="root",
            tokens_in=10,
            tokens_out=2,
            status_code=200,
            response_status="completed",
        )
        | call_updates
    )
    observations = join_agent_model_calls(observations, [call])
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    result = {
        "ng_agent_observations": observations.model_dump(mode="json"),
        "ng_trajectory": trajectory.model_dump(mode="json"),
        "ng_model_call_capture": {"calls": [call.model_dump(mode="json")]},
        "response": {"usage": {"input_tokens": 999, "output_tokens": 999}},
    }
    canonical = _build_trajectory_record(row, result)
    assert all(not turn.model_calls for turn in canonical.turns)
    assert canonical.invocations[0].model_calls[0].model_call_id == "captured"
    (tmp_path / "rollouts.jsonl").write_text(
        json.dumps(row | result | {"ng_trajectory": canonical.model_dump(mode="json")}) + "\n"
    )
    summary = run_health_checks(tmp_path / "rollouts.jsonl", workers=1)
    verdict = json.loads((tmp_path / "rollout_verdicts.jsonl").read_text())
    return canonical, summary.summary, verdict


@pytest.mark.parametrize(
    "parts",
    [
        [{"type": "text", "text": "answer"}],
        [{"type": "reasoning", "text": "thinking"}],
        [
            {
                "type": "tool",
                "tool": "bash",
                "callID": "tool",
                "state": {"status": "running", "input": {"command": "true"}},
            }
        ],
    ],
    ids=["answer", "reasoning-only", "tool-only"],
)
def test_sandbox_turns_enable_content_health_without_call_references(tmp_path, parse, parts):
    db = _session_db(tmp_path, [_policy(*parts)])
    trajectory, observations = _trajectory(parse, db)
    [turn] = trajectory.turns
    assert turn.timestamp == 1.0
    assert turn.turn_no == turn.step_count == 1
    assert turn.question is None
    assert turn.model_calls == []
    canonical, summary, verdict = _health(tmp_path, trajectory, observations)
    assert "turns_unavailable" not in {gap.code for gap in canonical.gaps}
    checks = {finding["check"] for finding in verdict["findings"]}
    assert not checks & {"agent_turn_hollow", "rollout_missing_agent_turns", "trajectory_capture_mismatch"}
    assert not {
        "agent_turn_hollow",
        "rollout_missing_agent_turns",
        "rollout_ended_on_failed_model_call",
        "model_call_zero_completion_tokens",
        "model_call_missing_token_counts",
        "trajectory_capture_mismatch",
        "model_call_runaway_generation",
    } & set(verdict["unobserved"])
    # Invocation ownership is sufficient to compare aggregate usage, even without turn links.
    assert "rollout_token_count_mismatch" not in verdict["unobserved"]
    mismatch = next(f for f in verdict["findings"] if f["check"] == "rollout_token_count_mismatch")
    assert mismatch["detail"] == {
        "transcript_prompt": 999,
        "transcript_completion": 999,
        "capture_prompt": 10,
        "capture_completion": 2,
    }
    assert summary["run"]["artifacts"]["coverage"]["task_no_successful_model_calls"]["unobserved"] == 1


def test_invocation_bound_failed_call_remains_evaluable(tmp_path, parse):
    trajectory, observations = _trajectory(parse, _session_db(tmp_path, [_policy({"type": "text", "text": "known"})]))
    _, _, verdict = _health(tmp_path, trajectory, observations, status_code=503, tokens_out=0)
    checks = {f["check"] for f in verdict["findings"]}
    assert "rollout_ended_on_failed_model_call" in checks
    # The 503 produced no generation, so it has no tokens to account for.
    assert not {"model_call_zero_completion_tokens", "model_call_missing_token_counts"} & checks
    assert not {"agent_turn_hollow", "rollout_missing_agent_turns"} & checks


@pytest.mark.parametrize("parts", [[], [{"type": "text", "text": ""}]])
def test_finished_empty_response_is_a_hollow_turn(tmp_path, parse, parts):
    trajectory, observations = _trajectory(parse, _session_db(tmp_path, [_policy(*parts)]))
    assert len(trajectory.turns) == 1
    _, _, verdict = _health(tmp_path, trajectory, observations)
    assert {"agent_turn_hollow", "rollout_missing_agent_turns"} <= {f["check"] for f in verdict["findings"]}


def test_sandbox_message_membership_numbering_and_synthetic_exclusion(tmp_path, parse):
    db = _session_db(
        tmp_path,
        [
            ("user", [{"type": "text", "text": "prompt"}]),
            _policy({"type": "text", "text": "one"}, {"type": "text", "text": "two"}, parentID="m0"),
            ("child", *_policy({"type": "reasoning", "text": "child thought"})),
            _policy({"type": "text", "text": "summary"}, summary=True),
            ("user", [{"type": "text", "text": "shell", "synthetic": True}]),
            ({"role": "assistant", "parentID": "m4"}, [{"type": "tool", "tool": "bash", "state": {}}]),
            ("user", [{"type": "subtask"}]),
            ({"role": "assistant", "parentID": "m6"}, [{"type": "tool", "tool": "task", "state": {}}]),
            _policy({"type": "text", "text": "last"}, {"type": "text", "text": "synthetic", "synthetic": True}),
        ],
        sessions=[("root", None), ("child", "root")],
    )
    trajectory, observations = _trajectory(parse, db)
    assert [(t.invocation_id, t.turn_no, t.step_count) for t in trajectory.turns] == [
        ("root", 1, 1),
        ("child", 1, 1),
        ("root", 2, 2),
    ]
    assert trajectory.turns[0].source_message_id == "m1"
    assert [message["content"][0]["text"] for message in trajectory.turns[0].answer] == ["one", "two"]
    assert all("id" not in message["content"][0] for message in trajectory.turns[0].answer)
    assert not trajectory.gaps
    assert len([record for record in observations.records if isinstance(record, AgentInvocation)]) == 2


@pytest.mark.parametrize(
    "message,parts",
    [
        ({"role": "assistant", "error": {"name": "Aborted"}}, []),
        ({"role": "assistant"}, [{"type": "step-start"}]),
        ({"role": "assistant"}, [{"type": "text", "text": "missing step evidence"}]),
        ({"role": "assistant"}, [{"type": "step-start"}, {"type": "text", "text": "partial"}]),
        ([], []),
        ({"unexpected": True}, []),
        ({"role": "unexpected"}, []),
    ],
)
def test_partial_evidence_preserves_known_turns_and_gates_content_checks(tmp_path, parse, message, parts):
    db = _session_db(tmp_path, [_policy({"type": "text", "text": "known"}), (message, parts)])
    trajectory, observations = _trajectory(parse, db)
    assert trajectory.turns[0].answer[0]["content"][0]["text"] == "known"
    _, _, verdict = _health(tmp_path, trajectory, observations)
    assert {"agent_turn_hollow", "rollout_missing_agent_turns"} <= set(verdict["unobserved"])
    assert "rollout_ended_on_failed_model_call" not in verdict["unobserved"]
    assert not {"agent_turn_hollow", "rollout_missing_agent_turns"} & {f["check"] for f in verdict["findings"]}


@pytest.mark.parametrize("bad_part", [None, [], {"type": []}, {"type": "text", "text": 4}, {"type": "file"}])
def test_malformed_parts_do_not_discard_other_turns(tmp_path, parse, bad_part):
    db = _session_db(tmp_path, [_policy({"type": "text", "text": "known"}, bad_part)])
    trajectory, _ = _trajectory(parse, db)
    assert len(trajectory.turns) == 1
    assert trajectory.gaps[0].code == "turns_unavailable"


@pytest.mark.parametrize(
    "timestamp,expected", [(None, 0.0), (False, 0.0), (-1, 0.0), ("bad", 0.0), (float("inf"), 0.0)]
)
def test_sandbox_timestamp_falls_back_to_database(tmp_path, parse, timestamp, expected):
    db = _session_db(tmp_path, [_policy(time={"created": timestamp})])
    trajectory, _ = _trajectory(parse, db)
    assert trajectory.turns[0].timestamp == expected


@pytest.mark.parametrize(
    "mutation",
    [
        "update message set time_created = NULL",
        "update part set session_id = 'other'",
        "update part set message_id = 'missing'",
        "update message set session_id = 'unknown'",
    ],
)
def test_unowned_or_untimed_artifact_does_not_invent_turns(tmp_path, parse, mutation):
    db = _session_db(tmp_path, [_policy(time={})])
    with sqlite3.connect(db) as con:
        con.execute(mutation)
    trajectory, _ = _trajectory(parse, db)
    assert trajectory.turns == []
    assert trajectory.gaps[0].code == "turns_unavailable"


def test_invalid_json_and_tool_state_are_partial_evidence(tmp_path, parse):
    db = _session_db(tmp_path, [_policy({"type": "tool", "tool": "bash", "state": "invalid"}, {"type": "patch"})])
    with sqlite3.connect(db) as con:
        con.execute("insert into part values ('bad-json', 'm0', 'root', '{', 20)")
    trajectory, _ = _trajectory(parse, db)
    assert len(trajectory.turns) == 1
    assert trajectory.turns[0].answer == []
    assert trajectory.gaps[0].code == "turns_unavailable"


@pytest.mark.parametrize("schema", ["1.0", "2.0"])
def test_activation_scoping_keeps_child_ownership_and_cross_activation_compaction(tmp_path, schema):
    from nemo_gym.trajectory import ContextCompactionObservation
    from responses_api_agents.opencode_agent.observability import scope_opencode_activation

    db = _session_db(
        tmp_path,
        [
            ("user", [{"type": "text", "text": "initial"}]),
            _policy(
                {
                    "type": "tool",
                    "tool": "task",
                    "callID": "spawn-child",
                    "state": {
                        "status": "completed",
                        "input": {"prompt": "inspect"},
                        "output": "done",
                        "metadata": {"sessionId": "child"},
                        "time": {"start": 1000, "end": 1500},
                    },
                },
                finish="stop",
            ),
            ("child", *_policy({"type": "text", "text": "child answer"}, finish="stop")),
            ("user", [{"type": "compaction", "auto": True, "tail_start_id": "m1"}]),
            _policy({"type": "text", "text": "summary"}, summary=True, parentID="m3", finish="stop"),
            _policy({"type": "text", "text": "correction"}, finish="stop"),
        ],
        sessions=[("root", None), ("child", "root")],
    )
    projected = []
    for index, ids in enumerate(({"m0", "m1", "m2"}, {"m3", "m4", "m5"}), 1):
        trajectory = TrajectoryRecord(task_id="task", rollout_id="episode")
        bundle = _parse_opencode_session(
            db,
            "fallback",
            trajectory,
            require_terminal_finish=True,
            message_ids=ids,
            observation_schema_version=schema,
        )
        scope_opencode_activation(
            bundle, trajectory, invocation_prefix=f"session:activation-{index}", activation_id=index
        )
        projected.append((bundle, trajectory))
    first, first_trajectory = projected[0]
    invocations = [r for r in first.records if isinstance(r, AgentInvocation)]
    child = next(r for r in invocations if r.harness_session_id == "child")
    assert child.parent_invocation_id == "session:activation-1:root"
    assert child.invocation_id == "session:activation-1:child" and child.activation_id == 1
    assert child.spawned_by_tool_call_id == ("call:p2" if schema == "2.0" else "spawn-child")
    assert {turn.activation_id for turn in first_trajectory.turns} == {1}
    second, second_trajectory = projected[1]
    assert [r.harness_session_id for r in second.records if isinstance(r, AgentInvocation)] == ["root"]
    [compaction] = [r for r in second.records if isinstance(r, ContextCompactionObservation)]
    assert compaction.invocation_id == "session:activation-2:root"
    assert compaction.first_kept_item_id == ("call:p2" if schema == "2.0" else "spawn-child")
    assert compaction.summary == "summary" and compaction.source_message_ids == ["m4"]
    assert all(gap.code != "compaction_first_kept_item_unavailable" for gap in second.gaps)
    assert [(turn.activation_id, turn.turn_no, turn.step_count) for turn in second_trajectory.turns] == [(2, 1, 1)]


@pytest.mark.parametrize("current", [False, True])
@pytest.mark.parametrize("role", ["assistant", "user"])
@pytest.mark.parametrize(
    ("kept", "model_ref", "first_kept"),
    [
        ("kept", None, "p2"),
        # A Gym reply's reasoning block is emitted first, as its own item, whichever activation parses it.
        ("<think>r</think>", ModelServerRef(type="responses_api_models", name="policy"), "p2-reasoning"),
        ("<think>r</think>\n\nkept", ModelServerRef(type="responses_api_models", name="policy"), "p2-reasoning"),
        ("<think>r</think>", None, "p2"),
    ],
    ids=["plain", "gym_reasoning_only", "gym_reply", "literal_think_text"],
)
def test_compaction_boundary_uses_only_emitted_assistant_items(tmp_path, current, role, kept, model_ref, first_kept):
    from nemo_gym.trajectory import ContextCompactionObservation

    db = _session_db(
        tmp_path,
        [
            (
                role,
                [{"type": "reasoning", "text": " "}, {"type": "text", "text": ""}, {"type": "text", "text": kept}],
            ),
            ("user", [{"type": "compaction", "tail_start_id": "m0"}]),
            _policy({"type": "text", "text": "summary"}, summary=True, parentID="m1", finish="stop"),
        ],
    )
    ids = {"m1", "m2"} | ({"m0"} if current else set())
    bundle = _parse_opencode_session(
        db, "fallback", model_ref=model_ref, message_ids=ids, observation_schema_version="2.0"
    )
    compaction = next(r for r in bundle.records if isinstance(r, ContextCompactionObservation))
    assert compaction.first_kept_item_id == (first_kept if role == "assistant" else None)
    if role == "assistant":
        assert all(gap.code != "compaction_first_kept_item_unavailable" for gap in bundle.gaps)


@pytest.mark.parametrize("persisted_step", [False, True])
def test_final_activation_preserves_native_step_or_explicit_gap(tmp_path, persisted_step):
    from nemo_gym.agents import AgentCloseSessionResponse
    from responses_api_agents.opencode_agent.observability import scope_opencode_activation
    from responses_api_agents.opencode_agent.tests.test_model_call_correlation import (
        POLICY,
        CaptureStore,
        _capture,
        _merge,
    )

    parts = [{"type": "text", "text": "private answer"}]
    if persisted_step:
        parts = [{"type": "step-start"}, *parts, {"type": "step-finish"}]
    db = _session_db(tmp_path, [({"role": "assistant", "time": {"created": 1000}}, parts)])
    trajectory, observations = _trajectory(
        partial(_parse_opencode_session, require_terminal_finish=True, message_ids={"m0"}, model_ref=POLICY), db
    )
    scope_opencode_activation(observations, trajectory, invocation_prefix="session:activation-8", activation_id=8)
    # Close serialization must retain evidence even when the invocation itself was interrupted.
    for invocation in observations.records:
        if isinstance(invocation, AgentInvocation):
            invocation.status = "incomplete"
            invocation.stop_reason = "session_budget_exhausted"
    close = AgentCloseSessionResponse(
        agent_session_id="session", agent_observations=observations, trajectory=trajectory, cleanup_confirmed=True
    )
    result = json.loads(close.model_dump_json())
    canonical = _build_trajectory_record(
        {"_ng_task_index": 0, "_ng_rollout_index": 0},
        {"ng_agent_observations": result["agent_observations"], "ng_trajectory": result["trajectory"]},
    )
    assert len(canonical.turns) == int(persisted_step)
    # A successful captured response is joined only when its native provider step exists.
    restored = AgentCloseSessionResponse.model_validate(result)
    store = CaptureStore(tmp_path / "capture")
    _capture(store, "final-call", "m0", session="session:activation-8:root")
    _, bundle, joined = _merge(store, restored.agent_observations, restored.trajectory)
    assert len(joined.turns) == int(persisted_step)
    if persisted_step:
        assert [call.model_call_id for call in joined.turns[0].model_calls] == ["final-call"]
    else:
        assert any(gap.code == "assistant_message_call_unmatched" for gap in bundle.gaps)
    if persisted_step:
        assert canonical.turns[0].source_message_id == "m0"
        assert canonical.turns[0].activation_id == 8
        assert all(gap.code != "turns_unavailable" for gap in canonical.gaps)
    else:
        gap = next(gap for gap in canonical.gaps if gap.code == "turns_unavailable")
        diagnostic = json.loads(gap.detail.split("native_capture_diagnostics=", 1)[1])
        assert diagnostic["messages"][0]["time"] == {"created": 1000}
        assert [part["type"] for part in diagnostic["parts"]] == ["text"]
        assert "private answer" not in gap.detail


def test_native_gap_diagnostics_are_bounded_and_content_free(tmp_path):
    db = _session_db(
        tmp_path,
        [({"role": "assistant", "time": {"created": 1000}}, [{"type": "text", "text": "secret"}] * 5)] * 40,
    )
    trajectory, _ = _trajectory(_parse_opencode_session, db)
    assert trajectory.turns == []
    gap = next(gap for gap in trajectory.gaps if gap.code == "turns_unavailable")
    diagnostic = json.loads(gap.detail.split("native_capture_diagnostics=", 1)[1])
    assert diagnostic["message_count"] == 40
    assert len(diagnostic["messages"]) == 32
    assert diagnostic["part_count"] == 160
    assert len(diagnostic["parts"]) == 128
    assert diagnostic["truncated"] is True
    assert "secret" not in gap.detail


async def test_v2_tool_artifacts_keep_partial_intervals_and_exact_request_joins(tmp_path):
    from nemo_gym.rollouts import enrich_run_result
    from nemo_gym.trajectory import ToolCallObservation

    def part(status, times, exit_code=None):
        return {
            "type": "tool",
            "tool": "bash",
            "callID": "reused",
            "state": {
                "status": status,
                "time": times,
                "input": {"command": "true"},
                "output": "ok" if status == "completed" else None,
                "metadata": {"exit": exit_code} if exit_code is not None else {},
            },
        }

    db = _session_db(
        tmp_path,
        [
            _policy(part("completed", {"start": 1000, "end": 2000}, 7)),
            _policy(part("running", {"start": 3000})),
        ],
    )
    trajectory = TrajectoryRecord(task_id="0", rollout_id="0-0")
    bundle = _parse_opencode_session(db, "0-0", trajectory, observation_schema_version="2.0")
    tools = [r for r in bundle.records if isinstance(r, ToolCallObservation)]
    assert len({r.span_id for r in tools}) == 2
    assert len({r.tool_call_id for r in tools}) == 2
    assert tools[0].native_tool_call_id == tools[1].native_tool_call_id == "reused"
    assert tools[0].status == "failed"
    assert tools[0].timings[0].duration_ms == 1000
    assert tools[1].status == "incomplete"
    assert tools[1].timings[0].started_at == 3
    assert tools[1].timings[0].completed_at is None
    assert tools[1].timings[0].duration_ms is None
    assert all(t.timings[0].boundary == "lifecycle" for t in tools)
    calls = [
        {
            "model_call_id": f"model-{index}",
            "client_session_id": "root",
            "client_assistant_message_id": f"m{index}",
            "response": {
                "output": [{"type": "function_call", "call_id": "reused", "name": "bash", "arguments": "{}"}]
            },
        }
        for index in range(2)
    ]
    result = {
        "ng_agent_observations": bundle.model_dump(mode="json"),
        "ng_trajectory": trajectory.model_dump(mode="json"),
        "ng_model_call_capture": {"calls": calls},
    }
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    await enrich_run_result(row, result, capture_dirs=[], observability_enabled=True, trajectory_schema_version="2.0")
    saved = result["ng_trajectory"]
    assert [t["model_call_id"] for t in saved["tool_calls"]] == ["model-0", "model-1"]
    assert [t["status"] for t in saved["tool_calls"]] == ["failed", "incomplete"]
    assert saved["turns"][0]["answer"][0]["call_id"] == saved["tool_calls"][0]["tool_call_id"]
    again = _parse_opencode_session(db, "0-0", observation_schema_version="2.0")
    assert [r.span_id for r in again.records if isinstance(r, ToolCallObservation)] == [r.span_id for r in tools]


@pytest.mark.parametrize(
    "times,required",
    [
        ({"start": 1000}, {"tool_completion_unavailable", "tool_clock_domain_unavailable"}),
        ({"start": 2000, "end": 1000}, {"tool_clock_discontinuity"}),
        ({"start": "invalid", "end": 3000}, {"tool_start_unavailable"}),
    ],
)
async def test_native_timing_gaps_follow_imported_attempt_through_replay(tmp_path, times, required):
    from nemo_gym.rollouts import enrich_run_result
    from nemo_gym.trajectory import ToolCallObservation

    db = _session_db(
        tmp_path,
        [
            _policy(
                {
                    "type": "tool",
                    "tool": "bash",
                    "callID": "native",
                    "state": {"status": "running", "input": {}, "time": times},
                }
            )
        ],
    )
    native = TrajectoryRecord(task_id="0", rollout_id="0-0")
    bundle = _parse_opencode_session(db, "0-0", native, agent_instance="agent", observation_schema_version="2.0")
    # The invocation supplies the same explicit attribution as the tool parser input.
    for record in bundle.records:
        if isinstance(record, AgentInvocation):
            record.agent_instance = "agent"
    (raw_tool,) = [record for record in bundle.records if isinstance(record, ToolCallObservation)]
    source_gaps = [gap for gap in bundle.gaps if gap.span_id == raw_tool.span_id]
    assert required <= {gap.code for gap in source_gaps}
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    result = {"ng_agent_observations": bundle.model_dump(mode="json")}
    first = None
    for _ in range(3):
        await enrich_run_result(
            row, result, capture_dirs=[], observability_enabled=True, trajectory_schema_version="2.0"
        )
        saved = TrajectoryRecord.model_validate(result["ng_trajectory"])
        (tool,) = saved.tool_calls
        for source in source_gaps:
            matched = [
                gap
                for gap in saved.gaps
                if gap.code == source.code and gap.span_id == tool.span_id and gap.detail == source.detail
            ]
            assert len(matched) == 1
            assert matched[0].agent_instance == tool.agent_instance == "agent"
            assert matched[0].invocation_id == tool.invocation_id
        assert all(gap.span_id != raw_tool.span_id for gap in saved.gaps)
        assert all(gap.span_id is None or gap.matches_tool(tool) for gap in saved.gaps)
        if first is not None:
            assert saved.tool_calls == first.tool_calls
            assert saved.gaps == first.gaps
        first = saved


async def test_persisted_part_ids_survive_growth_and_keep_native_primary_output(tmp_path):
    from nemo_gym.rollouts import enrich_run_result
    from nemo_gym.trajectory import ToolCallObservation
    from responses_api_agents.opencode_agent.artifacts import parse_opencode_session

    part = {
        "type": "tool",
        "callID": "c",
        "tool": "bash",
        "state": {"status": "running", "input": {}, "time": {"start": 1000}},
    }
    db = _session_db(tmp_path, [("assistant", [part])])
    bundle = _parse_opencode_session(db, "capture", observation_schema_version="2.0")
    result = {"ng_agent_observations": bundle.model_dump(mode="json")}
    row = {"task_id": "task", "_ng_rollout_id": "capture"}
    await enrich_run_result(row, result, capture_dirs=[], observability_enabled=True, trajectory_schema_version="2.0")
    initial = result["ng_trajectory"]["tool_calls"][0]
    assert initial["status"] == "incomplete"
    assert any(
        gap["code"] == "tool_completion_unavailable" and gap["span_id"] == initial["span_id"]
        for gap in result["ng_trajectory"]["gaps"]
    )
    part["state"].update(status="completed", output="done", metadata={"exit": 0}, time={"start": 1000, "end": 1010})
    with sqlite3.connect(db) as con:
        con.execute("update part set data=? where id='p0'", (json.dumps(part),))
        con.execute("insert into message values (?,?,?,?)", ("m1", "root", json.dumps({"role": "assistant"}), 1))
        con.execute("insert into part values (?,?,?,?,?)", ("p1", "m1", "root", json.dumps(part), 1))
        # Literal native ID that collided with the old conditional encoding.
        other = {**part, "callID": "c:p0"}
        con.execute("insert into part values (?,?,?,?,?)", ("p2", "m1", "root", json.dumps(other), 2))
    native_before = parse_opencode_session(db)
    full = _parse_opencode_session(db, "capture", observation_schema_version="2.0")
    sliced = _parse_opencode_session(db, "capture", message_ids={"m0"}, observation_schema_version="2.0")
    assert [tool.tool_call_id for tool in sliced.records if isinstance(tool, ToolCallObservation)] == [
        initial["tool_call_id"]
    ]
    result["ng_agent_observations"] = full.model_dump(mode="json")
    await enrich_run_result(row, result, capture_dirs=[], observability_enabled=True, trajectory_schema_version="2.0")
    tools = result["ng_trajectory"]["tool_calls"]
    assert len({tool["tool_call_id"] for tool in tools}) == 3
    assert tools[0]["span_id"] == initial["span_id"] and tools[0]["status"] == "completed"
    assert tools[0]["timings"][0]["duration_ms"] == 10
    assert not any(gap["code"] == "tool_attempt_identity_conflict" for gap in result["ng_trajectory"]["gaps"])
    assert [tool["native_tool_call_id"] for tool in tools] == ["c", "c", "c:p0"]
    assert parse_opencode_session(db) == native_before
    assert [item.call_id for item in native_before[0] if item.type == "function_call"] == ["c", "c", "c:p0"]


@pytest.mark.parametrize("schema", ["1.0", "2.0"])
def test_compaction_keeps_same_tool_reference_in_full_and_later_slice(tmp_path, schema):
    from nemo_gym.trajectory import ContextCompactionObservation, ToolCallObservation

    part = {
        "type": "tool",
        "callID": "reused",
        "tool": "bash",
        "state": {"status": "completed", "input": {}, "output": "done", "metadata": {"exit": 0}},
    }
    db = _session_db(
        tmp_path,
        [
            ("assistant", [part]),
            ("assistant", [part]),
            ("user", [{"type": "compaction", "auto": True, "tail_start_id": "m0"}]),
            _policy({"type": "text", "text": "summary"}, summary=True, parentID="m2", finish="stop"),
        ],
    )
    full = _parse_opencode_session(db, "capture", observation_schema_version=schema)
    sliced = _parse_opencode_session(db, "capture", message_ids={"m2", "m3"}, observation_schema_version=schema)
    first_id = next(record.tool_call_id for record in full.records if isinstance(record, ToolCallObservation))
    for bundle in (full, sliced):
        assert (
            next(
                record.first_kept_item_id
                for record in bundle.records
                if isinstance(record, ContextCompactionObservation)
            )
            == first_id
        )


async def test_missing_interrupted_artifact_reports_unknown_tool_coverage(tmp_path):
    from nemo_gym.rollouts import enrich_run_result, read_rollout_timing

    bundle = _parse_opencode_session(tmp_path / "missing.db", "capture", observation_schema_version="2.0")
    result = {"ng_agent_observations": bundle.model_dump(mode="json")}
    await enrich_run_result(
        {"task_id": "task", "_ng_rollout_id": "capture"},
        result,
        capture_dirs=[],
        observability_enabled=True,
        trajectory_schema_version="2.0",
    )
    assert result["ng_trajectory"]["tool_calls"] == []
    assert "agent_artifact_unavailable" in {gap["code"] for gap in result["ng_trajectory"]["gaps"]}
    assert read_rollout_timing(result).tool_s is None


@pytest.mark.parametrize("dialect", ["responses", "chat_completions"])
@pytest.mark.parametrize(
    "obstacle",
    [
        None,
        "message_id",
        "unproven_turn",
        "wrong_model",
        "missing_payload",
        "missing_history_payload",
        "wrong_qualifier",
        "reused_native",
    ],
)
async def test_native_projection_of_persisted_parts_and_captured_turns(tmp_path, dialect, obstacle):
    from copy import deepcopy

    from nemo_gym.rollouts import enrich_run_result
    from nemo_gym.trajectory import project_trajectory_v1

    part = {
        "type": "tool",
        "tool": "bash",
        "callID": "native",
        "state": {
            "status": "completed",
            "input": {"command": "true"},
            "output": "result",
            "time": {"start": 1000, "end": 2000},
        },
    }
    db = _session_db(
        tmp_path,
        [
            ("assistant", [part, *([deepcopy(part)] if obstacle == "reused_native" else [])]),
            ("assistant", [{"type": "text", "text": "done"}]),
        ],
    )
    bundle = _parse_opencode_session(db, "root", observation_schema_version="2.0")
    original = bundle.model_dump(mode="json")
    observed_call = original["records"][0]["conversation"][0]
    assert observed_call["id"] == observed_call["call_id"] == "call:p0"
    if dialect == "responses":
        answer = [
            {"type": "function_call", "id": "request-item", "call_id": "native", "name": "bash", "arguments": "{}"}
        ]
        history = [*answer, {"type": "function_call_output", "call_id": "native", "output": "result"}]
        request = {"input": [{"role": "user", "content": "run"}]}
        response = {"output": answer}
        later_request = {"input": history}
        later_response = {
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}]
        }
    else:
        answer = {
            "role": "assistant",
            "tool_calls": [{"id": "native", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
        }
        history = [answer, {"role": "tool", "tool_call_id": "native", "content": "result"}]
        request = {"messages": [{"role": "user", "content": "run"}]}
        response = {"choices": [{"message": answer}]}
        later_request = {"messages": history}
        later_response = {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
    calls = [
        ModelCallRecord(
            model_call_id=f"model-{index}",
            call_index=index,
            client_session_id="root",
            client_assistant_message_id=f"m{index}",
            status_code=200,
            response_status="completed",
            request=req,
            response=resp,
        ).model_dump(mode="json")
        for index, (req, resp) in enumerate([(request, response), (later_request, later_response)])
    ]
    original = join_agent_model_calls(bundle, [ModelCallRecord.model_validate(call) for call in calls]).model_dump(
        mode="json"
    )
    result = {
        "response": {"output": "primary"},
        "ng_agent_observations": original,
        "ng_model_call_capture": {"calls": calls},
    }
    fresh = deepcopy(result)
    await enrich_run_result(
        {"task_id": "task", "_ng_rollout_id": "capture"},
        result,
        capture_dirs=[],
        observability_enabled=True,
        trajectory_schema_version="2.0",
    )
    canonical = deepcopy(result["ng_trajectory"])
    assert canonical["tool_calls"][0]["model_call_id"] == "model-0"
    assert canonical["turns"][0]["model_calls"][0]["model_call_id"] == "model-0"
    assert canonical["turns"][1]["question"]
    if obstacle == "message_id":
        canonical["invocations"][0]["conversation"].append(
            {"type": "message", "id": "native", "role": "assistant", "content": []}
        )
    elif obstacle == "unproven_turn":
        canonical["turns"].append(
            {"invocation_id": "root", "answer": [{"type": "function_call", "call_id": "native"}], "model_calls": []}
        )
    elif obstacle == "wrong_model":
        canonical["turns"][0]["model_calls"] = [{"model_call_id": "model-1"}]
    elif obstacle == "missing_payload":
        canonical["model_calls"][0]["response"] = None
    elif obstacle == "missing_history_payload":
        canonical["model_calls"][1]["request"] = None
    elif obstacle == "wrong_qualifier":
        canonical["turns"][0]["model_calls"][0]["response_id"] = "wrong"
    before = deepcopy(canonical)
    projected = project_trajectory_v1(canonical)
    assert canonical == before
    assert projected["model_calls"][0]["response"] == canonical["model_calls"][0]["response"]
    assert [{k: v for k, v in turn.items() if k != "agent_instance"} for turn in canonical["turns"]] == projected[
        "turns"
    ]
    expected = "call:p0" if obstacle else "native"
    assert projected["tool_calls"][0]["tool_call_id"] == expected
    assert projected["invocations"][0]["conversation"][0]["call_id"] == expected
    assert projected["invocations"][0]["conversation"][0]["id"] == expected
    assert any(g["code"] == "tool_native_identity_unrepresentable" for g in projected["gaps"]) == bool(obstacle)
    if obstacle is None:
        for replay in (fresh, result):
            for _ in range(3):
                await enrich_run_result(
                    {"task_id": "task", "_ng_rollout_id": "capture"},
                    replay,
                    capture_dirs=[],
                    observability_enabled=True,
                )
                saved = TrajectoryRecord.model_validate(replay["ng_trajectory"])
                assert saved.tool_calls[0].tool_call_id == "native"
                assert replay["ng_agent_observations"]["records"][0]["conversation"][0]["id"] == "native"
                assert replay["response"] == {"output": "primary"}


def test_explicit_opencode_clock_domain_enables_overlapping_tool_busy_time(tmp_path):
    db = _session_db(
        tmp_path,
        [
            _policy(
                *[
                    {
                        "type": "tool",
                        "tool": "bash",
                        "callID": f"call-{index}",
                        "state": {
                            "status": "completed",
                            "input": {},
                            "output": "ok",
                            "time": {"start": start, "end": start + 1000},
                        },
                    }
                    for index, start in enumerate([1000, 1500])
                ]
            )
        ],
    )
    for domain, expected in [(None, None), ("opencode-wall:test", 1.5)]:
        bundle = _parse_opencode_session(db, "capture", observation_schema_version="2.0", clock_domain=domain)
        tools = [r.model_dump(mode="json") for r in bundle.records if r.kind == "tool_call"]
        assert all(t["timings"][0]["clock_domain"] == domain for t in tools)
        row = {"ng_trajectory": {"schema_version": "2.0", "capture_key": "capture", "tool_calls": tools}}
        assert read_rollout_timing(row).tool_s == expected
        replay = _parse_opencode_session(db, "capture", observation_schema_version="2.0", clock_domain=domain)
        assert replay == bundle
