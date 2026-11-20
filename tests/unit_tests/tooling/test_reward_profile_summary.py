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


def _fan_out_rows(runs, tasks=2, repeats=2):
    """The repeats of each task under each run, given as (agent, Environment Server or None)."""
    rows = []
    for task in range(tasks):
        for position, (agent, server) in enumerate(runs):
            for repeat in range(repeats):
                row = {
                    "_ng_task_index": task,
                    "_ng_rollout_index": repeats * position + repeat,
                    "agent_ref": {"name": agent},
                }
                rows.append(row | ({"_ng_environment_server": server} if server else {}))
    return rows


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
    # A smaller run's inputs plan fewer tasks or fewer repeats than the profile measured.
    for planned_rows in (rows[2:], rows[::2]):
        _write_jsonl(inputs, planned_rows)
        completed = _summarize(profile, check=False)
        assert completed.returncode != 0
        assert "do not match this profile (task 0)" in completed.stderr
    # Agent metrics from another run can count more masked rollouts than the inputs leave unmeasured.
    _write_jsonl(inputs, rows)
    metrics = tmp_path / "run_agent_metrics.json"
    saved_metrics = metrics.read_text()
    metrics.write_text(json.dumps([json.loads(saved_metrics)[0] | {"coverage/masked_rollouts": 1}]))
    completed = _summarize(profile, check=False)
    assert completed.returncode != 0
    assert "materialized inputs or the agent metrics do not match this profile" in completed.stderr
    metrics.write_text(saved_metrics)
    # The same tasks with twice the repeats, as a stale inputs file from an earlier run would have.
    extra = [row | {"_ng_rollout_index": row["_ng_rollout_index"] + 2} for row in rows]
    _write_jsonl(inputs, rows + extra)

    completed = _summarize(profile, check=False)

    assert completed.returncode != 0
    assert "materialized inputs or the agent metrics do not match this profile" in completed.stderr
    # Without the agent metrics to check against, the difference is reported as unknown rather than as masked.
    (tmp_path / "run_agent_metrics.json").unlink()
    assert _summary(profile)["coverage"]["masked_rollouts"] is None
    # A truncated profile is an error, not a smaller run.
    profile.write_text(profile.read_text()[:-20])
    completed = _summarize(profile, check=False)
    assert completed.returncode != 0
    assert "not valid JSON" in completed.stderr
    # Under several runs, inputs that name another run than a row's sample, or plan no run for main's count-only row
    # (task 1 saved nothing), do not match either.
    (tmp_path / "fan_out").mkdir()
    rows = _fan_out_rows([("agent_a", None), ("agent_b", None)])
    results = [row | {"reward": 1.0} for row in rows if row["_ng_task_index"] == 0]
    profile = _gym_profile(tmp_path / "fan_out", rows, results)
    renamed = [
        row | {"agent_ref": {"name": "agent_c"}} if row["agent_ref"]["name"] == "agent_b" else row for row in rows
    ]
    for planned_rows, task in [(renamed, 0), (rows[:4], 1)]:
        _write_jsonl(tmp_path / "fan_out" / "run_materialized_inputs.jsonl", planned_rows)
        completed = _summarize(profile, check=False)
        assert completed.returncode != 0
        assert f"do not match this profile (task {task})" in completed.stderr


def test_float_noise_is_not_spread(tmp_path):
    rows = [{"_ng_task_index": 0, "_ng_rollout_index": repeat, "agent_ref": {"name": "agent"}} for repeat in range(2)]
    profile = _gym_profile(tmp_path, rows, [rows[0] | {"reward": 0.1 + 0.2}, rows[1] | {"reward": 0.3}])

    assert "Same reward (0.3) on every repeat (no signal): 0\n" in _summarize(profile).stdout


def test_full_coverage_needs_no_masked_note(tmp_path):
    # Taskset rows name no agent, so the masked split is unknown, but nothing can be masked when all was measured.
    rows = [
        {"_ng_task_index": task, "_ng_rollout_index": repeat, "_ng_environment_server": "server"}
        for task in range(2)
        for repeat in range(2)
    ]
    profile = _gym_profile(tmp_path, rows, [row | {"reward": 1.0} for row in rows])

    assert _summarize(profile).stdout.startswith("Coverage: 4 of 4 planned rollouts measured (100.0%).\n")


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
    rows = _fan_out_rows(runs)
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


@pytest.mark.parametrize("shape", ["0.7.0", "main"])
def test_runs_renamed_by_masking_are_read_from_the_samples(tmp_path, shape):
    # gym names the runs that masking left. With server_b masked everywhere, the agent behind both servers is named by
    # its own name in the profile but by its servers in the inputs. A row with a measured rollout still shows its
    # server in `sample`, and main's count-only row for task 1 under server_a has the label of server_a's task 0 row.
    rows = _fan_out_rows([("agent", "server_a"), ("agent", "server_b"), ("agent_c", None)])
    results = [
        row | {"reward": 1.0, "mask_sample": row.get("_ng_environment_server") == "server_b"}
        for row in rows
        if (row["_ng_task_index"], row.get("_ng_environment_server")) != (1, "server_a")
    ]
    profile = _gym_profile(tmp_path, rows, results)
    if shape == "0.7.0":
        _drop_count_only_rows(profile)

    coverage = _summary(profile)["coverage"]

    assert (coverage["planned_rollouts"], coverage["measured_rollouts"]) == (12, 6)
    assert (coverage["masked_rollouts"], coverage["never_saved_rollouts"]) == (4, 2)
    assert _ids(coverage["tasks_without_measured_rollout"]) == [(0, "server_b"), (1, "server_a"), (1, "server_b")]


def test_a_count_only_row_takes_the_run_of_a_measured_row_with_its_label(tmp_path):
    # With server_b masked, gym labels server_a "agent", the inputs' label of the masked server named "agent". server_a
    # saved nothing for task 1, and its count-only row must not land on that server.
    rows = _fan_out_rows([("agent", "server_a"), ("agent", "server_b"), ("other", "agent"), ("other", "server_d")])
    results = [
        row | {"reward": 1.0, "mask_sample": row["_ng_environment_server"] in ("server_b", "agent")}
        for row in rows
        if (row["_ng_task_index"], row["_ng_environment_server"]) != (1, "server_a")
    ]
    profile = _gym_profile(tmp_path, rows, results)

    coverage = _summary(profile)["coverage"]

    assert (coverage["planned_rollouts"], coverage["measured_rollouts"]) == (16, 6)
    assert (coverage["masked_rollouts"], coverage["never_saved_rollouts"]) == (8, 2)
    absent = [(0, "agent"), (0, "server_b"), (1, "agent"), (1, "server_b")]
    assert _ids(coverage["tasks_not_in_profile_file"]) == absent
    assert _ids(coverage["tasks_without_measured_rollout"]) == sorted(absent + [(1, "server_a")])


def test_a_row_of_unknown_run_keeps_the_other_runs_coverage(tmp_path):
    # The inputs label server_a by its server, as its agent's name is the key of the server named "agent". With that
    # server and server_c masked, gym labels server_a "agent". server_a saved nothing, so no measured row tells which
    # run the count-only rows labelled "agent" belong to. Those runs stay unknown, and agent_d keeps its known
    # coverage: it measured 1 of task 0's 2 repeats.
    rows = _fan_out_rows([("agent", "server_a"), ("other", "agent"), ("other", "server_c"), ("agent_d", None)])
    results = [
        row | {"reward": 1.0, "mask_sample": row.get("_ng_environment_server") in ("agent", "server_c")}
        for row in rows
        if row.get("_ng_environment_server") != "server_a"
        and (row["_ng_task_index"], row["_ng_rollout_index"]) != (0, 7)
    ]
    profile = _gym_profile(tmp_path, rows, results)

    coverage = _summary(profile)["coverage"]

    assert (coverage["planned_rollouts"], coverage["measured_rollouts"]) == (16, 3)
    assert (coverage["masked_rollouts"], coverage["never_saved_rollouts"]) == (None, None)
    assert (coverage["tasks"], coverage["complete_tasks"]) == (8, 1)
    assert _ids(coverage["partial_tasks"]) == [(0, "agent_d")]
    assert _ids(coverage["tasks_without_measured_rollout"]) == [
        (task, server) for task in (0, 1) for server in ("agent", "server_a", "server_c")
    ]
    assert coverage["tasks_not_in_profile_file"] == []


def test_a_run_masked_everywhere_is_reported_apart(tmp_path):
    # With every agent_b result masked, gym sees a single run and writes a plain row for task 0.
    rows = _fan_out_rows([("agent_a", None), ("agent_b", None)], tasks=1)
    results = [row | {"reward": 1.0, "mask_sample": row["agent_ref"]["name"] == "agent_b"} for row in rows]
    profile = _gym_profile(tmp_path, rows, results)

    coverage = _summary(profile)["coverage"]

    assert (coverage["planned_rollouts"], coverage["measured_rollouts"]) == (4, 2)
    assert (coverage["masked_rollouts"], coverage["never_saved_rollouts"]) == (2, 0)
    assert (coverage["complete_tasks"], coverage["partial_tasks"]) == (1, [])
    assert _ids(coverage["tasks_without_measured_rollout"]) == [(0, "agent_b")]
    assert "  Not in the profile file (every saved rollout was masked): 0 (agent_b)\n" in _summarize(profile).stdout


@pytest.mark.parametrize(
    "runs", [[("agent_a", None), ("agent_b", None)], [("agent", "server_a"), ("agent", "server_b")]]
)
def test_without_inputs_a_run_that_saved_nothing_keeps_its_label(tmp_path, runs):
    # The second run saved nothing, so main writes its count-only row under the label gym gave it.
    rows = _fan_out_rows(runs, tasks=1)
    results = [row | {"reward": 1.0} for row in rows[:2]]
    profile = _gym_profile(tmp_path, rows, results)
    (tmp_path / "run_materialized_inputs.jsonl").unlink()
    measured, unmeasured = (server or agent for agent, server in runs)

    summary = _summary(profile)

    assert _ids(summary["same_reward_every_repeat"]["1.0"]) == [(0, measured)]
    assert _ids(summary["no_measured_rollout"]) == [(0, unmeasured)]
    assert summary["mean_reward_over_measured_rollouts"] == {measured: 1.0}
    assert summary["coverage"]["planned_rollouts"] is None


@pytest.mark.parametrize("shape", ["main", "0.7.0"])
def test_runs_on_separate_tasks_keep_their_own_means(tmp_path, shape):
    # Each task ran under one agent, so gym writes plain rows that do not name the run. Task 2 saved nothing.
    rows = [
        {"_ng_task_index": task, "_ng_rollout_index": repeat, "agent_ref": {"name": agent}}
        for task, agent in enumerate(["agent_a", "agent_b", "agent_b"])
        for repeat in range(2)
    ]
    results = [row | {"reward": float(row["agent_ref"]["name"] == "agent_a")} for row in rows[:4]]
    profile = _gym_profile(tmp_path, rows, results)
    if shape == "0.7.0":
        _drop_count_only_rows(profile)

    summary = _summary(profile)

    assert summary["mean_reward_over_measured_rollouts"] == {"agent_a": 1.0, "agent_b": 0.0}
    assert _ids(summary["coverage"]["tasks_without_measured_rollout"]) == [(2, "agent_b")]
    assert summary["coverage"]["never_saved_rollouts"] == 2
    # Without the inputs, the rows' samples still tell the runs apart.
    (tmp_path / "run_materialized_inputs.jsonl").unlink()
    assert _summary(profile)["mean_reward_over_measured_rollouts"] == {"agent_a": 1.0, "agent_b": 0.0}


def test_without_inputs_the_planned_total_is_unknown(tmp_path):
    profile = tmp_path / "math_rollouts_reward_profiling.jsonl"
    shutil.copy(SKILL / "evals" / "files" / "math_run" / profile.name, profile)

    summary = _summary(profile)

    assert summary["coverage"]["inputs_checked"] is False
    assert summary["coverage"]["planned_rollouts"] is None
    report = _summarize(profile).stdout
    assert "Materialized inputs not found" in report
    # nemo-gym 0.7.0 writes no row for tasks 12 and 14, so without the inputs they are unknown, not absent.
    assert "No measured rollout (not measured): none; tasks with no row in the profile file are unknown\n" in report
