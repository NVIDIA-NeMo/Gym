# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3
from functools import partial

import pytest

from nemo_gym.observability.model_calls import ModelCallRecord
from nemo_gym.rollout_collection import _build_trajectory_record
from nemo_gym.rollout_health import run_health_checks
from nemo_gym.trajectory import AgentInvocation, TrajectoryRecord
from nemo_gym.trajectory.trajectory_record import join_agent_model_calls
from responses_api_agents.opencode_agent.app import _parse_opencode_session
from responses_api_agents.opencode_agent.tests.test_app import _session_db


@pytest.fixture(
    params=[_parse_opencode_session, partial(_parse_opencode_session, require_terminal_finish=True)],
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
    assert trajectory.turns[0].answer[0]["id"] == "m1"
    assert [p["text"] for p in trajectory.turns[0].answer[0]["content"]] == ["one", "two"]
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


def test_activation_scoping_keeps_child_ownership_and_cross_activation_compaction(tmp_path):
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
        bundle = _parse_opencode_session(db, "fallback", trajectory, require_terminal_finish=True, message_ids=ids)
        scope_opencode_activation(
            bundle, trajectory, invocation_prefix=f"session:activation-{index}", activation_id=index
        )
        projected.append((bundle, trajectory))
    first, first_trajectory = projected[0]
    invocations = [r for r in first.records if isinstance(r, AgentInvocation)]
    child = next(r for r in invocations if r.harness_session_id == "child")
    assert child.parent_invocation_id == "session:activation-1:root"
    assert child.invocation_id == "session:activation-1:child" and child.activation_id == 1
    assert child.spawned_by_tool_call_id == "spawn-child"
    assert {turn.activation_id for turn in first_trajectory.turns} == {1}
    second, second_trajectory = projected[1]
    assert [r.harness_session_id for r in second.records if isinstance(r, AgentInvocation)] == ["root"]
    [compaction] = [r for r in second.records if isinstance(r, ContextCompactionObservation)]
    assert compaction.invocation_id == "session:activation-2:root"
    assert compaction.first_kept_item_id == "spawn-child"
    assert compaction.summary == "summary" and compaction.source_message_ids == ["m4"]
    assert all(gap.code != "compaction_first_kept_item_unavailable" for gap in second.gaps)
    assert [(turn.activation_id, turn.turn_no, turn.step_count) for turn in second_trajectory.turns] == [(2, 1, 1)]


@pytest.mark.parametrize("current", [False, True])
@pytest.mark.parametrize("role", ["assistant", "user"])
def test_compaction_boundary_uses_only_emitted_assistant_items(tmp_path, current, role):
    from nemo_gym.trajectory import ContextCompactionObservation

    db = _session_db(
        tmp_path,
        [
            (
                role,
                [{"type": "reasoning", "text": " "}, {"type": "text", "text": ""}, {"type": "text", "text": "kept"}],
            ),
            ("user", [{"type": "compaction", "tail_start_id": "m0"}]),
            _policy({"type": "text", "text": "summary"}, summary=True, parentID="m1", finish="stop"),
        ],
    )
    ids = {"m1", "m2"} | ({"m0"} if current else set())
    bundle = _parse_opencode_session(db, "fallback", message_ids=ids)
    compaction = next(r for r in bundle.records if isinstance(r, ContextCompactionObservation))
    assert compaction.first_kept_item_id == ("p2" if role == "assistant" else None)


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
