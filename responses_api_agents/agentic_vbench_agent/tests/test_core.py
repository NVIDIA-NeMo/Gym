# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from responses_api_agents.agentic_vbench_agent import core


def fixture_result(tmp_path, reward="0", status="OK", trajectory=True):
    job = tmp_path / "jobs/trial"
    (job / "agent").mkdir(parents=True)
    if trajectory:
        (job / "agent/trajectory.json").write_text(json.dumps({"steps": [{"source": "agent", "message": "done"}]}))
    (tmp_path / "agentic_vbench_results.tsv").write_text(
        f"task\tfamily\treward\tstatus\tjob_dir\nrepair1\tagentic_vbench_repair\t{reward}\t{status}\t{job}\n"
    )
    return {"task_id": "repair1", "family": "repair"}


def test_valid_zero_is_preserved(tmp_path):
    task = fixture_result(tmp_path)
    assert core.read_result(tmp_path, task)["reward"] == 0


@pytest.mark.parametrize("reward", ["nan", "inf", "-0.1", "1.1"])
def test_invalid_reward_fails(tmp_path, reward):
    task = fixture_result(tmp_path, reward=reward)
    with pytest.raises(ValueError, match="Invalid verifier"):
        core.read_result(tmp_path, task)


def test_infra_failure_does_not_become_zero(tmp_path):
    task = fixture_result(tmp_path, reward="", status="FAIL")
    with pytest.raises(RuntimeError, match="Unscored"):
        core.read_result(tmp_path, task)


def test_reward_without_model_trajectory_rejected(tmp_path):
    task = fixture_result(tmp_path, trajectory=False)
    with pytest.raises(RuntimeError, match="one model trajectory"):
        core.read_result(tmp_path, task)


def test_wrong_task_rejected(tmp_path):
    task = fixture_result(tmp_path)
    task["task_id"] = "different"
    with pytest.raises(ValueError, match="mismatched"):
        core.read_result(tmp_path, task)


def test_dataset_keeps_verbatim_prompt_and_no_media():
    prompt = "\nRead /workspace/input.mp4.\n\n"
    rows = core.dataset_rows({"x": {"task_id": "x", "family": "repair", "prompt": prompt}})
    assert rows[0]["responses_create_params"] == {"input": [{"role": "user", "content": prompt}]}
    assert "prompt" not in rows[0]["verifier_metadata"]
    with pytest.raises(ValueError):
        core.dataset_rows({}, "typo")
