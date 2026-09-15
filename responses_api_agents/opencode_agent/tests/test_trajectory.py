# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3

import pytest

from nemo_gym.base_responses_api_model import ModelCallRecord
from nemo_gym.rollout_collection import _build_trajectory_record
from nemo_gym.rollout_health import run_health_checks
from nemo_gym.rollout_observability import AgentInvocation, TrajectoryRecord, join_model_call_observations
from responses_api_agents.opencode_agent.app import _parse_opencode_session
from responses_api_agents.opencode_agent.tests.test_app import _session_db
from responses_api_agents.opencode_sandboxed_agent.app import parse_opencode_observations


@pytest.fixture(params=[_parse_opencode_session, parse_opencode_observations], ids=["local", "sandboxed"])
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
    observations = join_model_call_observations(observations, [call])
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
def test_native_turns_enable_content_health_without_call_references(tmp_path, parse, parts):
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
        "model_call_failed",
        "model_call_zero_completion_tokens",
        "model_call_missing_token_counts",
        "trajectory_capture_mismatch",
        "model_call_runaway_generation",
    } & set(verdict["unobserved"])
    assert "rollout_token_count_mismatch" in verdict["unobserved"]
    assert summary["run"]["artifacts"]["coverage"]["task_no_successful_model_calls"]["unobserved"] == 1


def test_invocation_bound_failed_call_remains_evaluable(tmp_path, parse):
    trajectory, observations = _trajectory(parse, _session_db(tmp_path, [_policy({"type": "text", "text": "known"})]))
    _, _, verdict = _health(tmp_path, trajectory, observations, status_code=503, tokens_out=0)
    assert {"model_call_failed", "model_call_zero_completion_tokens"} <= {f["check"] for f in verdict["findings"]}
    assert not {"agent_turn_hollow", "rollout_missing_agent_turns"} & {f["check"] for f in verdict["findings"]}


@pytest.mark.parametrize("parts", [[], [{"type": "text", "text": ""}]])
def test_finished_empty_response_is_a_hollow_turn(tmp_path, parse, parts):
    trajectory, observations = _trajectory(parse, _session_db(tmp_path, [_policy(*parts)]))
    assert len(trajectory.turns) == 1
    _, _, verdict = _health(tmp_path, trajectory, observations)
    assert {"agent_turn_hollow", "rollout_missing_agent_turns"} <= {f["check"] for f in verdict["findings"]}


def test_native_message_membership_numbering_and_synthetic_exclusion(tmp_path, parse):
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
    assert "model_call_failed" not in verdict["unobserved"]
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
def test_native_timestamp_falls_back_to_database(tmp_path, parse, timestamp, expected):
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
