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
"""Score a GDP.pdf run the way Artificial Analysis reports it.

Artificial Analysis (https://artificialanalysis.ai/methodology/intelligence-benchmarking#gdp-pdf)
attempts each of the 100 tasks five times and divides by all 500 attempts. Errors, missing attempts
and terminal input failures score zero, and a grading counts only when every criterion has a
verdict. Gym's own aggregate leaves failed rows out instead: a row whose judge call failed goes to
``<rollouts_stem>_failures.jsonl`` and drops out of the denominator.

This script reads the original verdicts (it calls no model or judge) and keeps every expected
task/attempt pair in the denominator. Recover judge failures first with
``gym eval reverify --judge-failed-only``; any attempt still without a verdict for every criterion
scores zero here and leaves ``judging_complete`` false. A run collected by ``gym eval`` is read through
its rollout store, so a re-judged or re-verified result replaces the one it settles.

    python benchmarks/gdp_pdf/report_aa.py --expected benchmarks/gdp_pdf/data/gdp_pdf_benchmark.jsonl \\
        --rollouts results/rollouts.jsonl --output results/gdp_pdf_aa_report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional, Sequence


AA_TASKS = 100
AA_REPEATS = 5
JUDGE_FAILED = "judge_failed"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def failures_path(rollouts_path: Path) -> Path:
    return rollouts_path.with_name(f"{rollouts_path.stem}_failures.jsonl")


def _attempt_key(row: dict[str, Any], task_ids_by_index: dict[Any, str]) -> Optional[tuple[str, int]]:
    task_id = (row.get("verifier_metadata") or {}).get("task_id")
    if task_id is None:
        task_id = task_ids_by_index.get(row.get("_ng_task_index"))
    repeat = row.get("_ng_rollout_index")
    if task_id is None or type(repeat) is not int:
        return None
    return str(task_id), repeat


def _answer_text(row: dict[str, Any]) -> str:
    """The final assistant text, which is all the judge sees (reasoning items are skipped)."""
    for item in reversed((row.get("response") or {}).get("output") or []):
        if item.get("type") == "message" and item.get("role") == "assistant":
            content = item.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                return "\n".join(c.get("text", "") for c in content if c.get("type") == "output_text").strip()
            return ""
    return ""


def _passes(score: Any) -> Optional[bool]:
    """The verifier's pass rule for a saved score (``str`` of a decimal; passes when it rounds half-up to 4 places
    to at least 1), or None when ``score`` is not a finite decimal string."""
    if not isinstance(score, str):
        return None
    try:
        value = Decimal(score)
        return value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP) >= 1 if value.is_finite() else None
    except InvalidOperation:
        return None


def _grade(row: dict[str, Any], criteria: dict[int, str]) -> Optional[tuple[float, float]]:
    """(all_pass, mean_pass) when every criterion of the expected rubric has a verdict, else None."""
    evaluations = row.get("criterion_evaluations") or []
    by_index = {item.get("index"): item for item in evaluations}
    if len(by_index) != len(evaluations) or by_index.keys() != criteria.keys():
        return None
    empty_answer = not _answer_text(row)
    for index, item in by_index.items():
        if item.get("criterion") != criteria[index]:
            raise ValueError(f"Judged criterion {index} differs from the expected rubric")
        if empty_answer:
            # The verifier skips the judge for an empty answer and fails every criterion.
            if item.get("passed") is not False:
                raise ValueError("An empty answer received criterion credit")
        elif type(item.get("passed")) is not bool or item["passed"] != _passes(item.get("score")):
            return None
    passed = sum(1 for item in evaluations if item["passed"])
    return float(passed == len(criteria)), passed / len(criteria)


def summarize(
    expected: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    failures: list[dict[str, Any]] = (),
    *,
    repeats: int = AA_REPEATS,
    inputs: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Aggregate a run over the fixed task-by-repeat denominator.

    ``inputs`` are the run's materialized input rows. They map a task index to its task, so a failure
    of a task with no successful rollout is still attributed to it.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    tasks: dict[str, dict[str, Any]] = {}
    for row in expected:
        metadata = row["verifier_metadata"]
        task_id = str(metadata["task_id"])
        if task_id in tasks and tasks[task_id]["criteria"] != metadata["criteria"]:
            raise ValueError(f"Task {task_id} appears twice with different rubrics")
        tasks[task_id] = metadata
    if not tasks:
        raise ValueError("No expected tasks")

    task_ids_by_index: dict[Any, str] = {}
    for row in [*inputs, *rollouts]:
        task_id = (row.get("verifier_metadata") or {}).get("task_id")
        if "_ng_task_index" in row and task_id is not None:
            if task_ids_by_index.setdefault(row["_ng_task_index"], str(task_id)) != str(task_id):
                raise ValueError(f"Task index {row['_ng_task_index']} maps to more than one task")
    graded_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rollouts:
        key = _attempt_key(row, task_ids_by_index)
        if key is None or key[0] not in tasks or not 0 <= key[1] < repeats:
            raise ValueError(f"Rollout outside the expected tasks and repeats: {key}")
        if key in graded_rows:
            raise ValueError(f"Duplicate rollout for task {key[0]} attempt {key[1]}")
        graded_rows[key] = row
    failure_classes: dict[tuple[str, int], str] = {}
    for row in failures:
        key = _attempt_key(row, task_ids_by_index)
        if key is not None and key not in graded_rows:
            failure_classes[key] = row.get("_ng_failure_class") or row.get("failure_kind") or "unknown"

    outcomes: Counter[str] = Counter()
    per_task = []
    domain_task_means: dict[str, list[float]] = defaultdict(list)
    for task_id, metadata in tasks.items():
        criteria = {item["index"]: item["criterion"] for item in metadata["criteria"]}
        if not criteria or len(criteria) != len(metadata["criteria"]):
            raise ValueError(f"Task {task_id} has an empty or duplicated rubric")
        all_pass_scores, mean_pass_scores = [], []
        for repeat in range(repeats):
            key = (task_id, repeat)
            row = graded_rows.get(key)
            grade = None
            if row is None:
                failure_class = failure_classes.get(key)
                outcomes[f"failed_{failure_class}" if failure_class else "missing"] += 1
            elif row.get("mask_sample") or row.get("failure_kind") == JUDGE_FAILED:
                outcomes[f"failed_{row.get('failure_kind') or 'masked'}"] += 1
            else:
                grade = _grade(row, criteria)
                if grade is None:
                    outcomes["incomplete_judging"] += 1
                elif row.get("failure_reason"):
                    outcomes["terminal_input_failure"] += 1
                elif not _answer_text(row):
                    outcomes["empty_answer"] += 1
                else:
                    outcomes["graded"] += 1
            all_pass, mean_pass = grade or (0.0, 0.0)
            all_pass_scores.append(all_pass)
            mean_pass_scores.append(mean_pass)
        task_mean_pass = sum(mean_pass_scores) / repeats
        domain_task_means[str(metadata.get("domain") or "unknown")].append(task_mean_pass)
        per_task.append(
            {
                "task_id": task_id,
                "domain": metadata.get("domain"),
                "all_pass": sum(all_pass_scores) / repeats,
                "mean_pass": task_mean_pass,
            }
        )

    denominator = len(tasks) * repeats
    # Zeros the verifier gave without the judge (an empty answer, an input that never fit) are final;
    # any other attempt without a complete grading is not judged yet.
    unjudged = denominator - sum(outcomes[name] for name in ("graded", "empty_answer", "terminal_input_failure"))
    return {
        "tasks": len(tasks),
        "repeats": repeats,
        "expected_attempts": denominator,
        "aa_shape": len(tasks) == AA_TASKS and repeats == AA_REPEATS,
        "all_pass": sum(task["all_pass"] for task in per_task) / len(per_task),
        "mean_pass": sum(task["mean_pass"] for task in per_task) / len(per_task),
        "domain_mean_pass": {domain: sum(v) / len(v) for domain, v in sorted(domain_task_means.items())},
        "attempt_outcomes": dict(sorted(outcomes.items())),
        # AA publishes a score only when every attempt has a verdict for every criterion. Missing and
        # failed attempts, and attempts whose grading failed, leave this false.
        "judging_complete": unjudged == 0,
        "all_attempts_present": outcomes["missing"] == 0,
        "per_task": per_task,
    }


def load_run(
    rollouts_path: Path, failures_fpath: Optional[Path] = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """(rollouts, failures, inputs) of a run.

    ``gym eval`` appends every outcome, including a re-judged or re-verified result after the one it
    settles, and records which one is current in a manifest. Such a run is read through its rollout
    store, which returns only the current outcome of each task/attempt pair and the run's materialized
    inputs. Rollouts without a manifest, or with an explicit failures file, are read as they are.
    """
    if failures_fpath is None:
        from nemo_gym.rollouts import RolloutStore

        store = RolloutStore.read(rollouts_path)
        if store is not None:
            rollouts, failures = store.selected("success"), store.failures()
            return rollouts, failures, store.inputs_for(rollouts + failures)
        failures_fpath = failures_path(rollouts_path)
    failures = read_jsonl(failures_fpath) if failures_fpath.is_file() else []
    return read_jsonl(rollouts_path), failures, []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--expected", type=Path, required=True, help="Prepared benchmark JSONL (one row per task)")
    parser.add_argument("--rollouts", type=Path, required=True, help="Scored rollouts JSONL")
    parser.add_argument(
        "--failures",
        type=Path,
        default=None,
        help="Failures sidecar of rollouts without a run manifest; giving one reads both files as they are "
        "(default: the run's rollout store, else <rollouts_stem>_failures.jsonl)",
    )
    parser.add_argument("--repeats", type=int, default=AA_REPEATS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rollouts, failures, inputs = load_run(args.rollouts, args.failures)
    report = summarize(read_jsonl(args.expected), rollouts, failures, repeats=args.repeats, inputs=inputs)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_task"}, indent=2))


if __name__ == "__main__":
    main()
