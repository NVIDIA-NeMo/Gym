# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy

import pytest

from benchmarks.swe_together.check_smoke import audit, inspect_row


def healthy(tmp_path):
    directory = tmp_path / "task"
    directory.mkdir()
    (directory / "canonical_goals.json").write_text(json.dumps({"completeness_goals": [{"id": "g", "weight": 1}]}))
    return {
        "task_id": "task",
        "reward": 0,
        "judge_score": 0,
        "verdict": {"goal_results": [{"id": "g", "met": False}]},
        "ng_activations": [
            {
                "activation_id": 1,
                "response": {
                    "metadata": {"harness_session_id": "native"},
                    "_ng_agent_observations": {"records": [{"kind": "agent_invocation", "activation_id": 1}]},
                },
            }
        ],
        "ng_agent_close": {"cleanup_confirmed": True},
        "judge_close": {
            "agent_session_id": "judge",
            "agent_observations": {"source": "claude_code"},
            "cleanup_confirmed": True,
        },
        "cleanup_receipts": {r: {"cleanup_confirmed": True} for r in ("candidate_sandbox", "judge_sandbox")},
        "submission": {"archive_sha256": "digest", "transfer": "shared_storage"},
        "intent_coverage": {
            key: 0.0 for key in ("coverage_rate", "weighted_coverage", "scope_precision", "overall_score")
        },
        "n_simulator_messages": 0,
        "user_correction": None,
    }


@pytest.mark.parametrize("judge_source", ["claude_code", "other_judge"])
def test_measured_zero_is_healthy_and_duplicates_are_not(tmp_path, judge_source):
    row = healthy(tmp_path)
    row["judge_close"]["agent_observations"]["source"] = judge_source
    assert inspect_row(row, asset_root=tmp_path) == []
    output = tmp_path / "rows.jsonl"
    output.write_text(json.dumps(row) + "\n")
    assert audit(output, asset_root=tmp_path, expected=["task"])["passed"]
    output.write_text((json.dumps(row) + "\n") * 2)
    report = audit(output, asset_root=tmp_path, expected=["task", "absent"])
    assert not report["passed"] and report["duplicates"] == ["task"] and report["missing"] == ["absent"]


@pytest.mark.parametrize(
    "field", ["verdict", "ng_activations", "ng_agent_close", "judge_close", "submission", "intent_coverage"]
)
def test_missing_measurement_evidence_fails(tmp_path, field):
    row = healthy(tmp_path)
    del row[field]
    assert inspect_row(row, asset_root=tmp_path)


def test_conversation_switch_fails_even_when_grading_succeeds(tmp_path):
    row = healthy(tmp_path)
    row["ng_activations"].append({"activation_id": 2, "response": {"metadata": {"harness_session_id": "other"}}})
    assert "native_conversation_not_continuous" in inspect_row(row, asset_root=tmp_path)


@pytest.mark.parametrize("activation", [0, 1])
@pytest.mark.parametrize("session_id", ["native", "other", None, "", 1, True, [], {}])
def test_every_activation_requires_the_same_conversation_identity(tmp_path, activation, session_id):
    row = healthy(tmp_path)
    second = deepcopy(row["ng_activations"][0])
    second["activation_id"] = 2
    second["response"]["_ng_agent_observations"]["records"][0]["activation_id"] = 2
    row["ng_activations"].append(second)
    metadata = row["ng_activations"][activation]["response"]["metadata"]
    if session_id is None:
        metadata.pop("harness_session_id")
    else:
        metadata["harness_session_id"] = session_id
    errors = inspect_row(row, asset_root=tmp_path)
    assert errors == ([] if session_id == "native" else ["native_conversation_not_continuous"])


def test_negative_judge_close_fails_even_with_sandbox_stop(tmp_path):
    row = healthy(tmp_path)
    row["judge_close"]["cleanup_confirmed"] = False
    assert "unconfirmed_judge_close" in inspect_row(row, asset_root=tmp_path)
