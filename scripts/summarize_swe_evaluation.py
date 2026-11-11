# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize raw SWE evaluation rows and compare policies on paired tasks.

Example: --run original EVALUATION.json trajectory_collection.jsonl --run trained
TRAINED_EVALUATION.json TRAINED_RESULTS.jsonl --baseline original --output summary.json.
This reads results only; it does not select checkpoints, submit runs, or alter inputs.
"""

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean


def _task_name(row: Mapping) -> str:
    identity = row.get("task_id")
    if not isinstance(identity, dict) or not isinstance(identity.get("task_id"), str):
        raise ValueError("Expected a materialized task_id with a string task_id field")
    return identity["task_id"]


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _percentile(values: list[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def task_bootstrap(values: Sequence[float], *, samples: int = 10000, seed: int = 42) -> list[float]:
    """Return a percentile 95% interval by resampling whole task values."""
    if not values or samples < 2:
        raise ValueError("Bootstrap requires tasks and at least two resamples")
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [_percentile(draws, 0.025), _percentile(draws, 0.975)]


def _outcome(reply: object, expected_task_id: dict) -> tuple[str, float | None, str | None, Counter]:
    observations = Counter()
    if not isinstance(reply, dict):
        return "invalid", None, "non_object_reply", observations
    if "_ng_failure_response_body" in reply or reply.get("_ng_failure_class"):
        return "infrastructure_incomplete", None, str(reply.get("_ng_failure_class") or "request_failed"), observations
    if reply.get("task_id") != expected_task_id or not isinstance(reply.get("episode_id"), dict):
        return "invalid", None, "reply_identity_mismatch", observations
    if reply.get("failure") is not None:
        failure = reply["failure"]
        reason = failure.get("failure_reason", "episode_failed") if isinstance(failure, dict) else "episode_failed"
        return "infrastructure_incomplete", None, str(reason), observations
    payload = reply.get("result")
    if not isinstance(payload, dict):
        return "invalid", None, "missing_verifier_result", observations
    agent_observations = payload.get("ng_agent_observations") or {}
    if isinstance(agent_observations, dict):
        for observation in agent_observations.get("records", []):
            if isinstance(observation, dict) and observation.get("error_type"):
                observations[str(observation["error_type"])] += 1
    completed = payload.get("evaluation_completed")
    if completed is not True:
        reason = "verification_incomplete" if completed is False else "verification_completion_missing"
        return "infrastructure_incomplete", None, reason, observations
    reward = payload.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        return "invalid", None, "nonfinite_or_missing_reward", observations
    if reward not in (0, 1):
        return "invalid", None, "nonbinary_swe_reward", observations
    return "completed", float(reward), None, observations


def summarize_run(name: str, evaluation_path: Path, results_path: Path, *, bootstrap_samples: int = 10000) -> dict:
    """Validate one full task×trial population; withhold scores until it is complete."""
    evaluation = json.loads(evaluation_path.read_text())
    tasks = _positive_int(evaluation["tasks"], "tasks")
    trials = _positive_int(evaluation["trials"], "trials")
    expected_count = _positive_int(evaluation["expected_episodes"], "expected_episodes")
    input_path = Path(evaluation["input"])
    input_bytes = input_path.read_bytes()
    expected = [json.loads(line) for line in input_bytes.decode().splitlines() if line.strip()]
    task_names = [_task_name(row) for row in expected]
    population = Counter(task_names)
    if len(expected) != expected_count or expected_count != tasks * trials:
        raise ValueError("Evaluation metadata and input row count disagree")
    if len(population) != tasks or set(population.values()) != {trials}:
        raise ValueError("Input must contain exactly the declared number of trials for every distinct task")
    # Trials of a task must use identical task contents, not merely a reused ID.
    task_inputs = {}
    for row, task in zip(expected, task_names):
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":"))
        if task in task_inputs and task_inputs[task] != encoded:
            raise ValueError(f"Input trials differ for task {task}")
        task_inputs[task] = encoded
    results_bytes = results_path.read_bytes() if results_path.exists() else b""
    errors = Counter()
    outcomes = Counter()
    failure_reasons = Counter()
    observation_errors = Counter()
    seen = set()
    task_rewards = defaultdict(list)
    parsed_rows = 0
    for line in results_bytes.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            errors["unparseable_row"] += 1
            continue
        parsed_rows += 1
        if not isinstance(record, dict):
            errors["non_object_row"] += 1
            continue
        index = record.get("evaluation_row_index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < expected_count:
            errors["invalid_row_index"] += 1
            continue
        if index in seen:
            errors["duplicate_row_index"] += 1
            continue
        seen.add(index)
        request = record.get("request")
        if not isinstance(request, dict) or request.get("task_id") != expected[index]["task_id"]:
            errors["request_identity_mismatch"] += 1
            outcomes["invalid"] += 1
            continue
        state, reward, reason, observations = _outcome(record.get("result"), expected[index]["task_id"])
        outcomes[state] += 1
        observation_errors.update(observations)
        if reason:
            failure_reasons[reason] += 1
        if reward is not None:
            task_rewards[task_names[index]].append(reward)
    missing = expected_count - len(seen)
    complete = not errors and not missing and outcomes["completed"] == expected_count
    status = "complete" if complete else "invalid" if errors or outcomes["invalid"] else "incomplete"
    means = {task: mean(values) for task, values in task_rewards.items()} if complete else None
    return {
        "name": name,
        "policy": evaluation["policy"],
        "step": evaluation["step"],
        "harness": evaluation["harness"],
        "evaluation_manifest": str(evaluation_path),
        "results": str(results_path),
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "result_snapshot_sha256": hashlib.sha256(results_bytes).hexdigest(),
        "status": status,
        "expected_tasks": tasks,
        "trials_per_task": trials,
        "expected_rows": expected_count,
        "parsed_rows": parsed_rows,
        "unique_row_indices": len(seen),
        "missing_rows": missing,
        "completed_verifications": outcomes["completed"],
        "infrastructure_incomplete_rows": outcomes["infrastructure_incomplete"],
        "invalid_rows": outcomes["invalid"],
        "invalid_record_counts": dict(errors),
        "failure_reasons": dict(failure_reasons),
        "agent_observation_error_counts": dict(observation_errors),
        "mean_single_attempt_success": mean(list(means.values())) if complete else None,
        "task_bootstrap_95_percent_interval": task_bootstrap(list(means.values()), samples=bootstrap_samples)
        if complete
        else None,
        "task_means": means,
        "score_definition": "Mean of all declared trials per task, then across tasks; not best-of-N.",
    }


def compare_runs(baseline: dict, candidate: dict, *, bootstrap_samples: int = 10000) -> dict:
    """Compare complete matched populations with a paired task bootstrap."""
    result = {"baseline": baseline["name"], "candidate": candidate["name"], "harness": candidate["harness"]}
    if baseline["harness"] != candidate["harness"] or baseline["input_sha256"] != candidate["input_sha256"]:
        return result | {"status": "not_comparable", "reason": "Different evaluation harness or frozen inputs"}
    if baseline["status"] != "complete" or candidate["status"] != "complete":
        return result | {
            "status": "incomplete",
            "mean_success_difference": None,
            "paired_task_bootstrap_95_percent_interval": None,
        }
    before, after = baseline["task_means"], candidate["task_means"]
    differences = [after[task] - before[task] for task in sorted(before)]
    return result | {
        "status": "complete",
        "paired_tasks": len(differences),
        "mean_success_difference": mean(differences),
        "paired_task_bootstrap_95_percent_interval": task_bootstrap(differences, samples=bootstrap_samples),
        "uncertainty": "Percentile bootstrap of paired task means, resampling tasks, not treating trials as independent tasks. No multiplicity correction; descriptive uncertainty only.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=3, action="append", required=True, metavar=("NAME", "EVALUATION", "RESULTS"))
    parser.add_argument(
        "--baseline", action="append", default=[], help="Run name; repeat for one baseline per harness"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    names = [run[0] for run in args.run]
    if len(set(names)) != len(names) or any(name not in names for name in args.baseline):
        parser.error("Run names must be unique and baselines must name supplied runs")
    runs = {name: summarize_run(name, Path(evaluation), Path(results)) for name, evaluation, results in args.run}
    comparisons = [
        compare_runs(runs[baseline], run)
        for baseline in args.baseline
        for name, run in runs.items()
        if name != baseline and run["harness"] == runs[baseline]["harness"]
    ]
    report = {
        "runs": list(runs.values()),
        "paired_comparisons": comparisons,
        "interpretation": "No score or interval is emitted for partial, invalid, or infrastructure-incomplete runs. Use internal validation for checkpoint selection; external test reports are final-only, never selection inputs.",
        "bootstrap_samples": 10000,
        "bootstrap_seed": 42,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
