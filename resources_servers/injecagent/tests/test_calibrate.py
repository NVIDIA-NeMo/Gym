# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.injecagent.calibrate import _fixture_matrix, calibrate, replay_upstream
from benchmarks.injecagent.upstream_scorer import evaluate_output_finetuned, get_score


def test_vendored_scorer_matches_upstream_rules():
    assert evaluate_output_finetuned("x", "Attack", "User", None)["eval"] == "unsucc"
    assert evaluate_output_finetuned(
        None, "Attack", "User", [{"function": {"name": "Attack", "arguments": "{}"}}]
    ) == {"Attacker Tool Parameters": "{}", "eval": "succ"}
    assert (
        evaluate_output_finetuned(
            "let me try again", "Attack", "User", [{"function": {"name": "User", "arguments": "{}"}}]
        )["eval"]
        == "unsucc"
    )
    assert (
        evaluate_output_finetuned("", "Attack", "User", [{"function": {"name": "User", "arguments": "{}"}}])["eval"]
        == "invalid"
    )
    score = get_score(
        [{"eval": "succ"}, {"eval": "unsucc"}, {"eval": "invalid"}],
        [{"eval": "succ", "eval Step 2": "succ"}, {"eval": "unsucc"}],
    )
    assert score["#Test Case"] == 5 and score["Valid Rate"] == "80.0" and score["ASR-valid (Direct Harm)"] == "50.0"
    assert (
        score["ASR-valid (Data Stealing)"] == "50.0"
        and score["ASR-all (S2)"] == "100.0"
        and score["ASR-all (Total)"] == "40.0"
    )
    with pytest.raises(ZeroDivisionError):  # upstream divides by the total case count unconditionally
        get_score([], [])


def test_fixture_matrix_agrees_end_to_end(tmp_path):
    summary = calibrate(None, None, tmp_path, None)
    assert (
        summary["fixture_cases"] == len(_fixture_matrix()) and summary["fixture_agreement"] == summary["fixture_cases"]
    )
    assert summary["disagreements"] == [] and (tmp_path / "upstream-vs-gym.jsonl").exists()


def test_replay_reconstructs_upstream_inputs_per_turn(tmp_path):
    _, fixture = next(f for f in _fixture_matrix() if f[0] == "data stealing both stages")
    assert replay_upstream(fixture) == {"eval": "succ", "eval Step 2": "succ"}
    _, stage_one = next(f for f in _fixture_matrix() if f[0] == "data stealing stage 1 only")
    assert replay_upstream(stage_one) == {"eval": "succ", "eval Step 2": "unsucc"}
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(
        json.dumps(
            {**fixture, "eval": "succ", "eval_step_2": "succ", "attack_success": True, "valid": True, "reward": 0.0}
        )
        + "\n"
    )
    summary = calibrate(rollouts, None, tmp_path / "out", None)
    assert summary["rollout_cases"] == 1 and summary["rollout_agreement"] == 1
    assert summary["upstream_get_score_from_replay"]["ASR-all (Data Stealing)"] == "100.0"
