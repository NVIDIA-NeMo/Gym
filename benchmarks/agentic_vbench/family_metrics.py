# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-family means and the equal-family mean of an Agentic-VBench rollouts file.

The benchmark's number is the unweighted mean of the four family means; Gym's aggregate metrics report the
task-weighted mean, so this reads the rows and groups them by the family each task's ``task_data.json`` names in the
entry folder the rows ran from. A masked row (``mask_sample``) is not a 0: it is counted apart from the scored rows.

    python -m benchmarks.agentic_vbench.family_metrics <rollouts.jsonl> [--entry environments/harbor/agentic_vbench]
"""

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from benchmarks.agentic_vbench.checkout import BENCHMARK_FAMILIES


def task_families(entry: Path) -> dict[str, str]:
    """Each task's family from its ``task_data.json`` in ``entry``."""
    families: dict[str, str] = {}
    for path in sorted(entry.glob("*/task_data.json")):
        families[path.parent.name] = json.loads(path.read_text())["family"]
    return families


def row_task_id(row: dict) -> str:
    """The task id of a results row (``_ng_task_id`` is the taskset-qualified id the harbor server writes)."""
    task_id = row.get("_ng_task_id")
    if isinstance(task_id, dict):
        return str(task_id["task_id"])
    if isinstance(task_id, str) and task_id:
        return task_id
    raise ValueError("a results row without _ng_task_id")


def family_metrics(rows: Iterable[dict], families: dict[str, str]) -> dict[str, dict[str, float | int]]:
    """Scored and masked counts and the mean reward of the scored rows, per family, plus ``equal_family`` over
    the families that have a scored row (``nan`` when none has)."""
    grouped: dict[str, dict[str, float | int]] = {
        family: {"rows": 0, "scored": 0, "masked": 0, "mean": math.nan} for family in sorted(BENCHMARK_FAMILIES)
    }
    sums: dict[str, float] = defaultdict(float)
    for row in rows:
        task_id = row_task_id(row)
        if task_id not in families:
            raise ValueError(f"task {task_id!r} has no task_data.json in the entry; prepare it first")
        family = families[task_id]
        group = grouped[family]
        group["rows"] += 1
        if row.get("mask_sample"):
            group["masked"] += 1
            continue
        reward = row.get("reward")
        if not isinstance(reward, (int, float)) or isinstance(reward, bool) or not math.isfinite(reward):
            raise ValueError(f"task {task_id!r}: a scored row without a finite reward ({reward!r})")
        group["scored"] += 1
        sums[family] += float(reward)
    for family, group in grouped.items():
        if group["scored"]:
            group["mean"] = sums[family] / group["scored"]
    means = [group["mean"] for group in grouped.values() if group["scored"]]
    grouped["equal_family"] = {"mean": sum(means) / len(means) if means else math.nan, "families": len(means)}
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "rollouts", type=Path, help="A rollouts.jsonl written by gym eval run or gym dataset validate."
    )
    parser.add_argument(
        "--entry",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "environments" / "harbor" / "agentic_vbench",
        help="The prepared entry folder the rows ran from (its task_data.json files name each task's family).",
    )
    args = parser.parse_args()
    with args.rollouts.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    metrics = family_metrics(rows, task_families(args.entry))
    for family in sorted(BENCHMARK_FAMILIES):
        group = metrics[family]
        print(
            f"{family:10s} rows={group['rows']:4d} scored={group['scored']:4d} masked={group['masked']:4d} "
            f"mean={group['mean']:.4f}"
        )
    equal = metrics["equal_family"]
    print(
        f"equal-family mean over {equal['families']} famil{'y' if equal['families'] == 1 else 'ies'}: {equal['mean']:.4f}"
    )
    if equal["families"] != len(BENCHMARK_FAMILIES):
        print("not the benchmark score: a family has no scored row")


if __name__ == "__main__":
    main()
