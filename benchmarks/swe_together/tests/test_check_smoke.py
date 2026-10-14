# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression checks for accepted zero scores and rejected continuation/grading evidence."""

import hashlib
import json
from copy import deepcopy

import pytest

from benchmarks.swe_together import check_smoke
from environment_servers.swe_together import task
from environment_servers.swe_together.coverage import compute_scores, normalize_match_table
from environment_servers.swe_together.evaluation import derive_score
from nemo_gym.agent_activation_types import AgentActivationResponse
from nemo_gym.base_responses_api_agent import AgentCloseSessionResponse
from nemo_gym.openai_utils import NeMoGymResponse


GOALS = {"completeness_goals": [{"id": "first", "weight": 0.75}, {"id": "second", "weight": 0.25}]}


def activation(index, *, dispatched=True):
    response = NeMoGymResponse(
        id=f"response-{index}",
        created_at=1,
        model="candidate",
        object="response",
        status="completed" if dispatched else "incomplete",
        output=[{"type": "message", "id": f"message-{index}", "content": [{"type": "output_text", "text": "Done."}]}]
        if dispatched
        else [],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        metadata={
            "native_conversation_id": "retained-conversation",
            "launch_attempted": str(dispatched).lower(),
            "native_input_accepted": str(dispatched).lower(),
            "policy_input_observed": "unknown",
        },
    )
    return AgentActivationResponse(
        activation_id=index,
        response=response,
        turn_complete=dispatched,
        stop_reason=None if dispatched else "session_budget_exhausted",
        observation={
            "events": [{"sequence": 0, "kind": "text", "text": "Done."}] if dispatched else [],
            "harness_steps": 1 if dispatched else 0,
            "duration_seconds": 1,
            "elapsed_seconds": index + 1,
        },
    ).model_dump(mode="json")


def attach_lifecycle(row, activations):
    row["ng_activations"] = activations
    row["ng_steps"] = [
        {
            "activation_id": index,
            "kind": "continue" if index < len(activations) - 1 else "stop",
            "responses_create_params": {"input": "Please continue."} if index < len(activations) - 1 else None,
            "stop_reason": None if index < len(activations) - 1 else item["stop_reason"] or "user_stopped",
        }
        for index, item in enumerate(activations)
    ]
    row["ng_agent_close"] = AgentCloseSessionResponse(
        agent_session_id="candidate-session", cleanup_confirmed=True
    ).model_dump(mode="json")
    # Keep malformed regression evidence intact; the auditor receives raw JSON.
    row["ng_agent_close"]["activations"] = deepcopy(activations)


@pytest.fixture
def healthy_zero():
    judge_close = AgentCloseSessionResponse(agent_session_id="judge-session", cleanup_confirmed=True).model_dump(
        mode="json"
    )
    verdict = derive_score(
        GOALS, {"goal_results": [{"id": goal["id"], "met": False} for goal in GOALS["completeness_goals"]]}
    )
    verdict["judge_close"] = judge_close
    row = {
        "task_id": "fixture",
        "reward": 0.0,
        "judge_score": 0.0,
        "verdict": verdict,
        "judge_close": deepcopy(judge_close),
        "cleanup_receipts": {
            "candidate_sandbox": {"cleanup_confirmed": True},
            "judge_sandbox": {"cleanup_confirmed": True},
        },
        "submission": {"archive_sha256": "a" * 64, "restored_roots": ["/workspace"], "members": 1},
        "n_simulator_messages": 0,
        "user_correction": 0.0,
        "intent_coverage": {
            "coverage_rate": 1.0,
            "weighted_coverage": 1.0,
            "scope_precision": 1.0,
            "overall_score": 1.0,
        },
    }
    attach_lifecycle(row, [activation(0)])
    return row


def test_real_contract_fixture_with_zero_solve_reward_is_healthy(healthy_zero):
    assert healthy_zero["judge_score"] == healthy_zero["reward"] == 0
    assert check_smoke.healthy_record(healthy_zero, goals=GOALS, n_intents=0) == []


@pytest.mark.parametrize("native_id", [None, "", "replacement-conversation"])
def test_resumption_rejects_missing_or_changed_native_conversation(healthy_zero, native_id):
    second = activation(1)
    second["response"]["metadata"]["native_conversation_id"] = native_id
    attach_lifecycle(healthy_zero, [activation(0), second])
    assert "missing_or_changed_native_conversation" in check_smoke.healthy_record(healthy_zero, goals=GOALS)


def test_harness_independent_resumption_needs_only_canonical_metadata(healthy_zero):
    attach_lifecycle(healthy_zero, [activation(0), activation(1)])
    assert check_smoke.healthy_record(healthy_zero, goals=GOALS) == []


def test_final_undispatched_budget_checkpoint_preserves_previous_conversation(healthy_zero):
    attach_lifecycle(healthy_zero, [activation(0), activation(1, dispatched=False)])
    assert check_smoke.healthy_record(healthy_zero, goals=GOALS) == []


def test_launched_but_unconfirmed_input_is_only_healthy_at_final_budget_stop(healthy_zero):
    checkpoint = activation(1, dispatched=False)
    checkpoint["response"]["metadata"].update(launch_attempted="true", native_input_accepted="unknown")
    attach_lifecycle(healthy_zero, [activation(0), checkpoint])
    assert check_smoke.healthy_record(healthy_zero, goals=GOALS) == []
    checkpoint["stop_reason"] = "activation_timeout"
    attach_lifecycle(healthy_zero, [activation(0), checkpoint])
    assert "invalid_unaccepted_budget_checkpoint" in check_smoke.healthy_record(healthy_zero, goals=GOALS)


def test_launch_does_not_prove_native_input_acceptance(healthy_zero):
    healthy_zero["ng_activations"][0]["response"]["metadata"]["native_input_accepted"] = "unknown"
    assert "invalid_unaccepted_budget_checkpoint" in check_smoke.healthy_record(healthy_zero, goals=GOALS)


@pytest.mark.parametrize("mutation", ["first", "changed_id", "output", "events", "usage", "continue"])
def test_rejects_invalid_undispatched_checkpoint(healthy_zero, mutation):
    checkpoint = activation(1, dispatched=False)
    if mutation == "changed_id":
        checkpoint["response"]["metadata"]["native_conversation_id"] = "replacement-conversation"
    elif mutation == "output":
        checkpoint["response"]["output"] = activation(0)["response"]["output"]
    elif mutation == "events":
        checkpoint["observation"]["events"] = activation(0)["observation"]["events"]
    elif mutation == "usage":
        checkpoint["response"]["usage"] = {"total_tokens": 1}
    attach_lifecycle(healthy_zero, [activation(0), checkpoint])
    if mutation == "first":
        checkpoint["activation_id"] = 0
        attach_lifecycle(healthy_zero, [checkpoint])
    elif mutation == "continue":
        healthy_zero["ng_steps"][-1].update(kind="continue", responses_create_params={"input": "again"})
    reasons = check_smoke.healthy_record(healthy_zero, goals=GOALS)
    assert {"invalid_unaccepted_budget_checkpoint", "missing_or_changed_native_conversation"} & set(reasons)


def test_false_goals_cannot_claim_perfect_score_even_with_consistent_fake_reward(healthy_zero):
    healthy_zero.update(judge_score=1.0, reward=1.0)
    healthy_zero["verdict"].update(judge_score=1.0, verdict="equivalent")
    assert "grade_disagrees_with_frozen_goals" in check_smoke.healthy_record(healthy_zero, goals=GOALS)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"id": "first", "met": False}],
        [{"id": "first", "met": False}, {"id": "first", "met": False}],
        [{"id": "first", "met": False}, {"id": "unknown", "met": False}],
        [{"id": "first", "met": False}, {"id": "second", "met": "false"}],
        [None],
    ],
)
def test_rejects_omitted_duplicate_unknown_or_nonboolean_goal_evidence(healthy_zero, rows):
    healthy_zero["verdict"]["goal_results"] = rows
    assert "invalid_frozen_goal_evidence" in check_smoke.healthy_record(healthy_zero, goals=GOALS)


@pytest.mark.parametrize("met,score,reward", [("first", 0.75, 0.0), ("both", 1.0, 1.0)])
def test_frozen_weights_and_production_reward_threshold(healthy_zero, met, score, reward):
    healthy_zero["verdict"]["goal_results"] = [
        {"id": goal["id"], "met": met == "both" or goal["id"] == met} for goal in GOALS["completeness_goals"]
    ]
    healthy_zero["verdict"] = derive_score(GOALS, healthy_zero["verdict"])
    healthy_zero.update(judge_score=score, reward=reward)
    assert check_smoke.healthy_record(healthy_zero, goals=GOALS) == []


@pytest.mark.parametrize("mutation", ["candidate_cleanup", "close_activations", "judge_cleanup", "judge_receipt"])
def test_requires_confirmed_and_matching_close_receipts(healthy_zero, mutation):
    if mutation == "candidate_cleanup":
        healthy_zero["ng_agent_close"]["cleanup_confirmed"] = False
    elif mutation == "close_activations":
        healthy_zero["ng_agent_close"]["activations"] = []
    elif mutation == "judge_cleanup":
        healthy_zero["verdict"]["judge_close"]["cleanup_confirmed"] = False
    else:
        healthy_zero["judge_close"]["agent_session_id"] = "different-judge"
    assert any("close" in reason for reason in check_smoke.healthy_record(healthy_zero, goals=GOALS))


@pytest.mark.parametrize("role", ["candidate_sandbox", "judge_sandbox"])
def test_agent_close_does_not_replace_owned_sandbox_cleanup(healthy_zero, role):
    healthy_zero["cleanup_receipts"][role]["cleanup_confirmed"] = False
    assert "unconfirmed_" + role + "_close" in check_smoke.healthy_record(healthy_zero, goals=GOALS)


def test_grade_without_frozen_filesystem_transfer_is_not_qualified(healthy_zero):
    del healthy_zero["submission"]
    assert "missing_frozen_submission_receipt" in check_smoke.healthy_record(healthy_zero, goals=GOALS)


@pytest.mark.parametrize("mutation", ["activation_order", "step_kind", "step_order", "event_order"])
def test_rejects_broken_activation_and_next_user_turn_order(healthy_zero, mutation):
    attach_lifecycle(healthy_zero, [activation(0), activation(1)])
    if mutation == "activation_order":
        healthy_zero["ng_activations"].reverse()
    elif mutation == "step_kind":
        healthy_zero["ng_steps"][0]["kind"] = "stop"
    elif mutation == "step_order":
        healthy_zero["ng_steps"].reverse()
    else:
        healthy_zero["ng_activations"][0]["observation"]["events"] *= 2
    assert check_smoke.healthy_record(healthy_zero, goals=GOALS)


def test_metrics_recompute_tag_corrections_and_coverage_match_evidence(healthy_zero):
    healthy_zero.update(
        n_simulator_messages=1,
        user_correction=1.2,
        message_tags=[{"trial_idx": 1, "tags": ["correction", "nudge"]}],
    )
    table = {
        "n_intents": 1,
        "n_trial_msgs": 2,
        "per_intent": [{"intent_id": 0, "matched_trial_idx": 1, "match_confidence": 0.8}],
    }
    healthy_zero["intent_coverage"] = compute_scores(table, 1, 2) | {"match_table": table}
    assert check_smoke.healthy_record(healthy_zero, goals=GOALS, n_intents=1) == []
    healthy_zero["user_correction"] = 0
    healthy_zero["intent_coverage"]["overall_score"] = 1
    assert {"missing_or_invalid_user_correction", "intent_coverage_disagrees_with_evidence"} <= set(
        check_smoke.healthy_record(healthy_zero, goals=GOALS, n_intents=1)
    )


@pytest.mark.parametrize("missing_evidence", [True, False])
def test_missing_intent_evidence_is_not_a_measured_zero(healthy_zero, missing_evidence):
    evidence = (
        {}
        if missing_evidence
        else {
            "per_intent": [
                {"intent_id": index, "matched_trial_idx": None, "match_confidence": 0.0} for index in range(2)
            ]
        }
    )
    table, warnings = normalize_match_table(evidence, 2, 1)
    healthy_zero["intent_coverage"] = compute_scores(table, 2, 1) | {
        "match_table": table,
        "schema_warnings": warnings,
    }
    assert healthy_zero["intent_coverage"]["overall_score"] == 0.0
    problems = check_smoke.healthy_record(healthy_zero, goals=GOALS, n_intents=2)
    assert problems == (["missing_or_invalid_intent_coverage"] if missing_evidence else [])


@pytest.mark.parametrize("mutation", ["missing_tags", "missing_match_table", "wrong_oracle_count", "nan", "bool"])
def test_rejects_missing_or_invalid_interaction_metrics(healthy_zero, mutation):
    n_intents = 0
    if mutation == "missing_tags":
        healthy_zero["n_simulator_messages"] = 1
    elif mutation == "missing_match_table":
        n_intents = 1
    elif mutation == "wrong_oracle_count":
        n_intents = 2
        table = {"n_intents": 0, "n_trial_msgs": 1, "per_intent": []}
        healthy_zero["intent_coverage"] = compute_scores(table, 0, 1) | {"match_table": table}
    else:
        healthy_zero["intent_coverage"]["overall_score"] = float("nan") if mutation == "nan" else True
    assert check_smoke.healthy_record(healthy_zero, goals=GOALS, n_intents=n_intents)


@pytest.fixture
def assets(tmp_path, monkeypatch):
    directory = tmp_path / "source/tasks/fixture"
    directory.mkdir(parents=True)
    (directory / "task.toml").write_text('[environment]\ndocker_image = "ghcr.io/official/task:pin"\n')
    (directory / "canonical_goals.json").write_text(json.dumps(GOALS))
    (directory / "oracle_intents.json").write_text('{"intents": []}')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "fixture",
                        "files": {
                            file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in directory.iterdir()
                        },
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(task, "MANIFEST", manifest)
    data = task.TaskData(task_id="fixture", image="ghcr.io/official/task:pin", image_digest="sha256:" + "a" * 64)
    return tmp_path / "source", {"fixture": data.model_dump()}


def write_results(path, *rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_retry_selection_is_first_healthy_in_file_order_and_retains_failures(healthy_zero, assets, tmp_path):
    source, data = assets
    failed = deepcopy(healthy_zero)
    failed["mask_sample"] = True
    first = write_results(tmp_path / "retry-b.jsonl", failed, healthy_zero, healthy_zero)
    second = write_results(tmp_path / "retry-a.jsonl", healthy_zero)
    report = check_smoke.audit([first, second], data, task_source=source)
    assert report["passed"] is True
    assert report["attempts"] == 4
    assert report["problem_counts"] == {"failed_or_masked": 1}
    assert report["selected_attempts"]["fixture"]["file"] == str(first)
    assert report["selected_attempts"]["fixture"]["line"] == 2
    assert report["attempt_records"][0]["problems"] == ["failed_or_masked"]


@pytest.mark.parametrize("wrapper", ["envelope", "collector"])
def test_audit_accepts_episode_envelope_and_flattened_collector_rows(healthy_zero, assets, tmp_path, wrapper):
    source, data = assets
    identity = {"taskset": "swe_together:full109", "task_id": "fixture"}
    row = (
        {"task_id": identity, "result": healthy_zero, "failure": None}
        if wrapper == "envelope"
        else (healthy_zero | {"_ng_task_id": identity})
    )
    path = write_results(tmp_path / "results.jsonl", row)
    assert check_smoke.audit([path], data, task_source=source)["passed"] is True


def test_audit_rejects_wrong_task_identity_and_missing_healthy_task(healthy_zero, assets, tmp_path):
    source, data = assets
    healthy_zero["task_id"] = "wrong-task"
    row = {"task_id": {"task_id": "fixture"}, "result": healthy_zero}
    path = write_results(tmp_path / "results.jsonl", row)
    report = check_smoke.audit([path], data, task_source=source)
    assert report["passed"] is False
    assert report["tasks_without_healthy_grade"] == ["fixture"]
    assert report["problem_counts"] == {"task_identity_mismatch": 1}


def test_audit_keeps_malformed_or_unexpected_rows_visible(healthy_zero, assets, tmp_path):
    source, data = assets
    path = write_results(tmp_path / "results.jsonl", healthy_zero, {"task_id": ["unexpected"]})
    with path.open("a") as output:
        output.write("not-json\n")
    report = check_smoke.audit([path], data, task_source=source)
    assert report["passed"] is False
    assert report["problem_counts"] == {"unexpected_task": 2}
    assert report["attempts"] == 3


def test_audit_checks_pinned_canonical_assets_before_grading(healthy_zero, assets, tmp_path):
    source, data = assets
    (source / "tasks/fixture/canonical_goals.json").write_text('{"completeness_goals": []}')
    path = write_results(tmp_path / "results.jsonl", healthy_zero)
    with pytest.raises(ValueError, match="asset is missing or changed"):
        check_smoke.audit([path], data, task_source=source)
