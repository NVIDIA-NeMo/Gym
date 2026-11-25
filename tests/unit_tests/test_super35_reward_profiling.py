# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/nemotron_3.5_super/reward_profiling.py"
filter_inputs = runpy.run_path(str(SCRIPT))["filter_inputs"]


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def result(task: int, repeat: int, reward: float, **kwargs) -> dict:
    return {"_ng_task_index": task, "_ng_rollout_index": repeat, "reward": reward, **kwargs}


def test_cli_filters_original_inputs_with_exclusive_bounds_and_shuffled_results(tmp_path):
    # Dataset idx is deliberately unrelated to Gym's task index.
    inputs = write_jsonl(tmp_path / "inputs.jsonl", [{"idx": 50 - i, "task_source": "swe"} for i in range(4)])
    rewards = [[0, 0, 0, 1], [0, 1, 0, 1], [0, 1, 1, 1], [1, 1, 1, 1]]
    rows = [
        result(task, repeat, reward) for task, values in enumerate(rewards) for repeat, reward in enumerate(values)
    ]
    rollouts = write_jsonl(tmp_path / "rollouts.jsonl", list(reversed(rows)))
    write_jsonl(
        tmp_path / "rollouts_materialized_inputs.jsonl",
        [{key: row[key] for key in ("_ng_task_index", "_ng_rollout_index")} for row in rows],
    )
    output = tmp_path / "nested" / "filtered"
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-jsonl",
            str(inputs),
            "--rollouts-jsonl",
            str(rollouts),
            "--output-dir",
            str(output),
            "--low-threshold",
            "0.25",
            "--high-threshold",
            "0.75",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert output.joinpath("filtered_inputs.jsonl").read_text() == inputs.read_text().splitlines(keepends=True)[1]
    assert [row["decision"] for row in read_jsonl(output / "reward_profile.jsonl")] == [
        "rejected_low",
        "kept",
        "rejected_high",
        "rejected_high",
    ]
    summary = json.loads(output.joinpath("summary.json").read_text())
    assert summary["kept_tasks"] == 1
    assert summary["rejected_low_tasks"] == 1
    assert summary["rejected_high_tasks"] == 2
    assert "Filtered dataset:" in process.stdout
    for stage in range(1, 6):
        assert f"[{stage}/5]" in process.stderr
    assert "kept_tasks_provisional=1" in process.stderr
    assert "Done: kept 1/4 input rows (1/4 tasks)" in process.stderr


def test_stable_task_ids_and_nonbinary_rewards(tmp_path):
    rows = [{"_ng_task_index": 90, "task": "first"}, {"_ng_task_index": 12, "task": "second"}]
    inputs = write_jsonl(tmp_path / "inputs.jsonl", rows)
    rollouts = write_jsonl(
        tmp_path / "rollouts.jsonl", [result(12, 0, -4), result(90, 0, 2), result(90, 1, 6), result(12, 1, 0)]
    )
    output = tmp_path / "output"
    filter_inputs(inputs, rollouts, output, num_repeats=2, low_threshold=0, high_threshold=5)
    assert read_jsonl(output / "filtered_inputs.jsonl") == rows[:1]
    assert [row["mean_reward"] for row in read_jsonl(output / "reward_profile.jsonl")] == [4, -2]


def test_duplicate_input_lines_share_gym_task_id_and_preserve_input_multiplicity(tmp_path):
    rows = [{"task": "duplicate"}, {"task": "duplicate"}, {"task": "last"}]
    inputs = write_jsonl(tmp_path / "inputs.jsonl", rows)
    rollouts = write_jsonl(tmp_path / "rollouts.jsonl", [result(0, 0, 0), result(0, 1, 1), result(1, 0, 1)])
    write_jsonl(tmp_path / "rollouts_materialized_inputs.jsonl", read_jsonl(rollouts))
    output = tmp_path / "output"
    summary = filter_inputs(inputs, rollouts, output)
    assert read_jsonl(output / "filtered_inputs.jsonl") == rows[:2]
    assert summary["input_tasks"] == 2
    assert summary["kept_rows"] == 2
    assert summary["kept_tasks"] == 1


def test_masks_and_inconclusive_results_do_not_become_zero_reward(tmp_path):
    inputs = write_jsonl(tmp_path / "inputs.jsonl", [{"task": "measured"}, {"task": "unmeasured"}])
    rollouts = write_jsonl(
        tmp_path / "rollouts.jsonl",
        [
            result(0, 0, 0.5),
            result(0, 1, 0, mask_sample=True),
            result(0, 2, 0, evaluation_completed=False),
            result(1, 0, 0, mask_sample=True),
            result(1, 1, 0, mask_sample=True),
            {"_ng_task_index": 1, "_ng_rollout_index": 2, "evaluation_completed": False},
        ],
    )
    output = tmp_path / "output"
    summary = filter_inputs(inputs, rollouts, output, num_repeats=3)
    assert read_jsonl(output / "filtered_inputs.jsonl") == [{"task": "measured"}]
    profiles = read_jsonl(output / "reward_profile.jsonl")
    assert profiles[0]["mean_reward"] == 0.5
    assert profiles[0]["measured_rollouts"] == 1
    assert profiles[1]["mean_reward"] is None
    assert profiles[1]["decision"] == "no_measured_rewards"
    assert summary["excluded_rollouts"] == 5


def test_missing_rollouts_require_opt_in_and_are_reported(tmp_path):
    inputs = write_jsonl(tmp_path / "inputs.jsonl", [{"task": "partial"}, {"task": "missing"}])
    rollouts = write_jsonl(tmp_path / "rollouts.jsonl", [result(0, 0, 0.5)])
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "filtered_inputs.jsonl"
    sentinel.write_text("previous output\n")
    with pytest.raises(ValueError, match="Incomplete rollouts for 2 tasks"):
        filter_inputs(inputs, rollouts, output, num_repeats=2)
    assert sentinel.read_text() == "previous output\n"
    summary = filter_inputs(inputs, rollouts, output, num_repeats=2, allow_partial_rollouts=True)
    assert summary["tasks_with_missing_rollouts"] == 2
    assert read_jsonl(sentinel) == [{"task": "partial"}]
    assert [row["missing_rollouts"] for row in read_jsonl(output / "reward_profile.jsonl")] == [1, 2]


def test_unknown_repeat_count_is_not_silently_assumed_complete(tmp_path):
    inputs = write_jsonl(tmp_path / "inputs.jsonl", [{"task": "a"}])
    rollouts = write_jsonl(tmp_path / "rollouts.jsonl", [result(0, 4, 0.5)])
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="supply --num-repeats"):
        filter_inputs(inputs, rollouts, output)
    filter_inputs(inputs, rollouts, output, allow_partial_rollouts=True)
    assert read_jsonl(output / "reward_profile.jsonl")[0]["expected_rollouts"] is None


@pytest.mark.parametrize(
    "rows,match",
    [
        ([result(0, 0, 0), result(0, 0, 1)], "duplicate rollout identity"),
        ([result(9, 0, 0.5)], "absent from the input dataset"),
        ([result(0, 2, 0.5)], "absent from the expected collection"),
        ([result(0, 0, float("nan"))], "finite number"),
        ([result(0, 0, float("inf"))], "finite number"),
        ([result(0, 0, "0.5")], "finite number"),
        ([{"_ng_task_index": 0, "_ng_rollout_index": 0}], "finite number"),
        ([{"reward": 0.5}], "_ng_task_index"),
        ([result(True, 0, 0.5)], "_ng_task_index"),
        ([result(0, 0, 0.5, agent_ref={"name": "a"}), result(0, 1, 0.5, agent_ref={"name": "b"})], "multiple agents"),
    ],
)
def test_bad_rollouts_fail_before_writing(tmp_path, rows, match):
    inputs = write_jsonl(tmp_path / "inputs.jsonl", [{"task": "a"}])
    rollouts = write_jsonl(tmp_path / "rollouts.jsonl", rows)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match=match):
        filter_inputs(inputs, rollouts, output, num_repeats=2)
    assert not output.exists()


@pytest.mark.parametrize("low,high", [(0.5, 0.5), (1, 0), (float("nan"), 1), (0, float("inf"))])
def test_invalid_thresholds(tmp_path, low, high):
    with pytest.raises(ValueError, match="thresholds"):
        filter_inputs(
            tmp_path / "input", tmp_path / "rollouts", tmp_path / "output", low_threshold=low, high_threshold=high
        )


def test_output_cannot_overwrite_source(tmp_path):
    inputs = write_jsonl(tmp_path / "filtered_inputs.jsonl", [{"task": "a"}])
    rollouts = write_jsonl(tmp_path / "rollouts.jsonl", [result(0, 0, 0.5)])
    with pytest.raises(ValueError, match="must not overwrite"):
        filter_inputs(inputs, rollouts, tmp_path, num_repeats=1)
    assert read_jsonl(inputs) == [{"task": "a"}]
