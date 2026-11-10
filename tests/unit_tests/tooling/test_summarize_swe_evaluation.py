# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from pathlib import Path

import pytest
from scripts.summarize_swe_evaluation import compare_runs, summarize_run


def make_run(tmp_path: Path, name: str, rewards: list[int]) -> tuple[Path, Path, list[dict]]:
    inputs = [
        {"task_id": {"taskset": "swe_next_opencode", "task_id": str(i // 3)}, "task_input": {}} for i in range(6)
    ]
    input_path = tmp_path / "inputs.jsonl"
    input_path.write_text("".join(json.dumps(row) + "\n" for row in inputs))
    evaluation = tmp_path / f"{name}.json"
    evaluation.write_text(
        json.dumps(
            {
                "policy": name,
                "step": 0,
                "harness": "opencode",
                "tasks": 2,
                "trials": 3,
                "expected_episodes": 6,
                "input": str(input_path),
            }
        )
    )
    records = [
        {
            "evaluation_row_index": i,
            "request": row,
            "result": {
                "task_id": row["task_id"],
                "episode_id": {"capture_key": f"rollout-{i}"},
                "result": {"evaluation_completed": True, "reward": reward},
            },
        }
        for i, (row, reward) in enumerate(zip(inputs, rewards))
    ]
    results = tmp_path / f"{name}.jsonl"
    write_records(results, records)
    return evaluation, results, records


def write_records(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_mean_uses_all_attempts_and_pairs_whole_tasks(tmp_path: Path) -> None:
    baseline, before, _ = make_run(tmp_path, "baseline", [0, 0, 0, 1, 1, 1])
    candidate, after, _ = make_run(tmp_path, "candidate", [1, 0, 0, 1, 1, 1])
    b = summarize_run("baseline", baseline, before, bootstrap_samples=100)
    c = summarize_run("candidate", candidate, after, bootstrap_samples=100)
    assert b["status"] == c["status"] == "complete"
    assert c["mean_single_attempt_success"] == pytest.approx(2 / 3)
    assert c["task_means"] == {"0": 1 / 3, "1": 1}
    comparison = compare_runs(b, c, bootstrap_samples=1000)
    assert comparison["paired_tasks"] == 2
    assert comparison["mean_success_difference"] == pytest.approx(1 / 6)
    assert comparison["paired_task_bootstrap_95_percent_interval"] == [0, 1 / 3]


@pytest.mark.parametrize("kind", ["partial", "infrastructure", "duplicate", "identity", "malformed", "nan"])
def test_incomplete_or_invalid_results_never_emit_a_score(tmp_path: Path, kind: str) -> None:
    evaluation, results, records = make_run(tmp_path, "test", [1] * 6)
    if kind == "partial":
        records.pop()
    elif kind == "infrastructure":
        records[0]["result"]["result"]["evaluation_completed"] = False
    elif kind == "duplicate":
        records.append(deepcopy(records[0]))
    elif kind == "identity":
        records[0]["result"]["task_id"] = {"task_id": "wrong"}
    elif kind == "nan":
        records[0]["result"]["result"]["reward"] = float("nan")
    write_records(results, records)
    if kind == "malformed":
        with results.open("a") as output:
            output.write('{"unfinished":')
    summary = summarize_run("test", evaluation, results, bootstrap_samples=10)
    assert summary["status"] != "complete"
    assert summary["mean_single_attempt_success"] is None
    assert summary["task_bootstrap_95_percent_interval"] is None
    if kind == "partial":
        assert summary["missing_rows"] == 1
    if kind == "infrastructure":
        assert summary["infrastructure_incomplete_rows"] == 1
        assert summary["completed_verifications"] == 5


def test_tool_errors_are_not_confused_with_incomplete_verification(tmp_path: Path) -> None:
    evaluation, results, records = make_run(tmp_path, "test", [0] * 6)
    records[0]["result"]["result"]["ng_agent_observations"] = {"records": [{"error_type": "tool_error"}]}
    write_records(results, records)
    summary = summarize_run("test", evaluation, results, bootstrap_samples=10)
    assert summary["status"] == "complete"
    assert summary["mean_single_attempt_success"] == 0
    assert summary["agent_observation_error_counts"] == {"tool_error": 1}


def test_sidecar_and_episode_failures_keep_expected_denominator(tmp_path: Path) -> None:
    evaluation, results, records = make_run(tmp_path, "test", [1] * 6)
    records[0]["result"] = {"_ng_failure_response_body": {"status": 503}}
    records[1]["result"].pop("result")
    records[1]["result"]["failure"] = {"failure_reason": "sandbox_failed"}
    write_records(results, records)
    summary = summarize_run("test", evaluation, results, bootstrap_samples=10)
    assert summary["status"] == "incomplete"
    assert summary["expected_rows"] == 6
    assert summary["infrastructure_incomplete_rows"] == 2
    assert summary["failure_reasons"] == {"request_failed": 1, "sandbox_failed": 1}


def test_comparison_refuses_different_harness_or_population(tmp_path: Path) -> None:
    evaluation, results, _ = make_run(tmp_path, "test", [1] * 6)
    summary = summarize_run("test", evaluation, results, bootstrap_samples=10)
    assert compare_runs(summary, summary | {"harness": "pi"})["status"] == "not_comparable"
    assert compare_runs(summary, summary | {"input_sha256": "different"})["status"] == "not_comparable"
    assert compare_runs(summary, summary | {"status": "incomplete"})["mean_success_difference"] is None


def test_trial_population_is_checked_before_scoring(tmp_path: Path) -> None:
    evaluation, results, _ = make_run(tmp_path, "test", [1] * 6)
    input_path = Path(json.loads(evaluation.read_text())["input"])
    rows = [json.loads(line) for line in input_path.read_text().splitlines()]
    rows[0]["task_id"]["task_id"] = "new-task"
    write_records(input_path, rows)
    with pytest.raises(ValueError, match="trials"):
        summarize_run("test", evaluation, results)
