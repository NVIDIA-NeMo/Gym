# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SKILL = Path(__file__).resolve().parents[3] / "skills" / "nemo-gym-reward-profiling"
SCRIPT = SKILL / "scripts" / "summarize_profile.py"
GYM = "import sys; from nemo_gym.cli.main import main; sys.argv = ['gym', *sys.argv[1:]]; main()"


def _summarize(profile, *args, check=True):
    # -S: the skill promises the script runs from a plain Python install, without nemo-gym or other packages.
    return subprocess.run(
        [sys.executable, "-S", str(SCRIPT), str(profile), *args],
        check=check,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _summary(profile):
    return json.loads(_summarize(profile, "--json").stdout)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _gym_profile(tmp_path, rows, results):
    """Profile the rows and results with `gym eval profile` itself and return the per-task profile path."""
    _write_jsonl(tmp_path / "run_materialized_inputs.jsonl", rows)
    _write_jsonl(tmp_path / "run.jsonl", results)
    subprocess.run(
        [sys.executable, "-c", GYM, "eval", "profile", "--inputs", "run_materialized_inputs.jsonl"]
        + ["--rollouts", "run.jsonl", "++allow_partial_rollouts=True"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=180,
    )
    return tmp_path / "run_reward_profiling.jsonl"


def _drop_count_only_rows(profile):
    # nemo-gym 0.7.0 writes no row for a task with no measured rollout; main writes one with num_rollouts 0.
    rows = [json.loads(line) for line in profile.read_text().splitlines()]
    _write_jsonl(profile, [row for row in rows if row["num_rollouts"]])


def _ids(entries):
    return sorted((entry["task"], entry["agent"]) for entry in entries)


def test_report_on_the_eval_math_run():
    # The eval cases quote these numbers; the run was profiled with nemo-gym 0.7.0, so tasks 12 and 14 have no row.
    report = _summarize(SKILL / "evals" / "files" / "math_run" / "math_rollouts_reward_profiling.jsonl").stdout

    assert report.splitlines()[:4] == [
        "Coverage: 98 of 128 planned rollouts measured (76.56%): 30 never saved, 0 masked.",
        "Tasks: 16; 11 complete, 3 partial, 2 with no measured rollout.",
        "  Not in the profile file (no saved rollout): 12, 14",
        "Mean reward over measured rollouts: 0.5510 (says nothing about tasks with none).",
    ]
    assert "Reward varies across repeats (learning signal): 6, 7, 8, 9, 10, 11*\n" in report
    assert "Same reward (0.0) on every repeat (no signal): 4, 5, 15*\n" in report
    assert "Same reward (1.0) on every repeat (no signal): 0, 1, 2, 3\n" in report
    assert "Only one measured rollout (spread unknown): 13*\n" in report


@pytest.mark.parametrize("rows_name_agent", [True, False])
def test_masked_rollouts_are_saved_but_not_measured(tmp_path, rows_name_agent):
    # Rows of Environment Server tasksets name no agent, and gym then reports no masked total in the agent metrics.
    run = {"agent_ref": {"name": "agent"}} if rows_name_agent else {"_ng_environment_server": "server"}
    rows = [{"_ng_task_index": task, "_ng_rollout_index": repeat} | run for task in range(3) for repeat in range(4)]
    # Every planned rollout was saved. Task 1 is masked on every repeat and task 0 on one, so gym writes no row for
    # task 1 and counts 3 of 4 for task 0.
    results = [row | {"reward": 1.0, "mask_sample": row["_ng_task_index"] == 1 or row == rows[0]} for row in rows]
    profile = _gym_profile(tmp_path, rows, results)

    coverage = _summary(profile)["coverage"]
    assert (coverage["planned_rollouts"], coverage["measured_rollouts"]) == (12, 7)
    assert _ids(coverage["partial_tasks"]) == [(0, None)]
    assert _ids(coverage["tasks_not_in_profile_file"]) == [(1, None)]
    if rows_name_agent:
        assert (coverage["masked_rollouts"], coverage["never_saved_rollouts"]) == (5, 0)
        assert "  Not in the profile file (every saved rollout was masked): 1\n" in _summarize(profile).stdout
    else:
        assert (coverage["masked_rollouts"], coverage["never_saved_rollouts"]) == (None, None)
        assert (
            "  Not in the profile file (never saved, or every saved rollout masked): 1\n" in _summarize(profile).stdout
        )

    if rows_name_agent:
        inputs = tmp_path / "run_materialized_inputs.jsonl"
        saved_inputs = inputs.read_text()
        # Without the inputs, the masked total cannot be checked against the planned rollouts.
        inputs.unlink()
        coverage = _summary(profile)["coverage"]
        assert (coverage["masked_rollouts"], coverage["masked_rollouts_in_agent_metrics"]) == (None, 5)
        # Without the agent metrics, the masked rollouts of task 1 cannot be told from rollouts never saved.
        inputs.write_text(saved_inputs)
        (tmp_path / "run_agent_metrics.json").unlink()
        assert _summary(profile)["coverage"]["never_saved_rollouts"] is None


def test_inputs_from_another_run_are_rejected(tmp_path):
    rows = [
        {"_ng_task_index": task, "_ng_rollout_index": repeat, "agent_ref": {"name": "agent"}}
        for task in range(2)
        for repeat in range(2)
    ]
    profile = _gym_profile(tmp_path, rows, [row | {"reward": 1.0} for row in rows])
    inputs = tmp_path / "run_materialized_inputs.jsonl"
    # A smaller run's inputs plan fewer rollouts than the profile measured.
    _write_jsonl(inputs, rows[2:])
    completed = _summarize(profile, check=False)
    assert completed.returncode != 0
    assert "do not match this profile (task 0)" in completed.stderr
    # The same tasks with twice the repeats, as a stale inputs file from an earlier run would have.
    extra = [row | {"_ng_rollout_index": row["_ng_rollout_index"] + 2} for row in rows]
    _write_jsonl(inputs, rows + extra)

    completed = _summarize(profile, check=False)

    assert completed.returncode != 0
    assert "do not match this profile" in completed.stderr
    # Without the agent metrics to check against, the difference is reported as unknown rather than as masked.
    (tmp_path / "run_agent_metrics.json").unlink()
    assert _summary(profile)["coverage"]["masked_rollouts"] is None
    # A truncated profile is an error, not a smaller run.
    profile.write_text(profile.read_text()[:-20])
    completed = _summarize(profile, check=False)
    assert completed.returncode != 0
    assert "not valid JSON" in completed.stderr


def test_float_noise_is_not_spread(tmp_path):
    rows = [{"_ng_task_index": 0, "_ng_rollout_index": repeat, "agent_ref": {"name": "agent"}} for repeat in range(2)]
    profile = _gym_profile(tmp_path, rows, [rows[0] | {"reward": 0.1 + 0.2}, rows[1] | {"reward": 0.3}])

    assert "Same reward (0.3) on every repeat (no signal): 0\n" in _summarize(profile).stdout


def test_long_task_lists_are_cut_in_the_text_report(tmp_path):
    rows = [
        {"_ng_task_index": task, "_ng_rollout_index": repeat, "agent_ref": {"name": "agent"}}
        for task in range(25)
        for repeat in range(2)
    ]
    profile = _gym_profile(tmp_path, rows, [row | {"reward": float(row["_ng_rollout_index"])} for row in rows])

    listed = ", ".join(str(task) for task in range(20))
    assert f"Reward varies across repeats (learning signal): {listed}, and 5 more (see --json)\n" in (
        _summarize(profile).stdout
    )
    assert len(_summary(profile)["varies"]) == 25


@pytest.mark.parametrize(
    "runs, labels",
    [
        pytest.param([("agent_a", None), ("agent_b", None)], ["agent_a", "agent_b"], id="agents"),
        pytest.param([("agent_a", "server_a"), ("agent_b", "server_b")], ["agent_a", "agent_b"], id="one_server_each"),
        pytest.param([("agent", "server_a"), ("agent", "server_b")], ["server_a", "server_b"], id="shared_agent"),
    ],
)
def test_fan_out_gap_for_one_run_counts_in_either_version(tmp_path, runs, labels):
    # gym names a run by its agent, unless several Environment Servers front that agent.
    rows = []
    for task in range(2):
        for position, (agent, server) in enumerate(runs):
            for repeat in range(2):
                row = {
                    "_ng_task_index": task,
                    "_ng_rollout_index": 2 * position + repeat,
                    "agent_ref": {"name": agent},
                }
                rows.append(row | ({"_ng_environment_server": server} if server else {}))
    # The second run saved nothing for task 1, which the first run did save.
    second_run = runs[1][1] or runs[1][0]
    gap = (1, labels[1])
    results = [
        row | {"reward": float(row["_ng_rollout_index"] % 2)}
        for row in rows
        if (row["_ng_task_index"], row.get("_ng_environment_server") or row["agent_ref"]["name"]) != (1, second_run)
    ]
    profile = _gym_profile(tmp_path, rows, results)

    for shape in ("main", "0.7.0"):
        if shape == "0.7.0":
            _drop_count_only_rows(profile)
        summary = _summary(profile)
        coverage = summary["coverage"]
        assert (coverage["planned_rollouts"], coverage["measured_rollouts"], coverage["never_saved_rollouts"]) == (
            8,
            6,
            2,
        ), shape
        assert _ids(coverage["tasks_without_measured_rollout"]) == [gap], shape
        # One mean per run, named the way gym labels the runs.
        assert sorted(summary["mean_reward_over_measured_rollouts"]) == labels, shape


def test_run_labels_that_change_with_masking_leave_the_split_unknown(tmp_path):
    # gym labels runs from the measured rows only. With server_b masked everywhere, the agent behind both servers
    # is labelled by its name in the profile but by its servers in the inputs, so masked and never-saved rollouts
    # cannot be told apart from these files.
    runs = [("agent", "server_a"), ("agent", "server_b"), ("agent_c", None)]
    rows = []
    for task in range(2):
        for position, (agent, server) in enumerate(runs):
            for repeat in range(2):
                row = {
                    "_ng_task_index": task,
                    "_ng_rollout_index": 2 * position + repeat,
                    "agent_ref": {"name": agent},
                }
                rows.append(row | ({"_ng_environment_server": server} if server else {}))
    results = [
        row | {"reward": 1.0, "mask_sample": row.get("_ng_environment_server") == "server_b"}
        for row in rows
        if (row["_ng_task_index"], row["agent_ref"]["name"]) != (0, "agent_c")
    ]
    profile = _gym_profile(tmp_path, rows, results)
    _drop_count_only_rows(profile)

    coverage = _summary(profile)["coverage"]

    assert (coverage["planned_rollouts"], coverage["measured_rollouts"]) == (12, 6)
    assert (coverage["masked_rollouts"], coverage["never_saved_rollouts"]) == (None, None)


def test_without_inputs_the_planned_total_is_unknown(tmp_path):
    profile = tmp_path / "math_rollouts_reward_profiling.jsonl"
    shutil.copy(SKILL / "evals" / "files" / "math_run" / profile.name, profile)

    summary = _summary(profile)

    assert summary["coverage"]["inputs_checked"] is False
    assert summary["coverage"]["planned_rollouts"] is None
    assert "Materialized inputs not found" in _summarize(profile).stdout
