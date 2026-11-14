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
"""ARC-AGI-2 scores from Gym rollouts, the way ARC Prize counts them.

A task is scored per test input: an input counts as solved when one of two attempts matches every cell, and the task's
score is the fraction of its test inputs solved. The benchmark score is the mean task score over the 120 tasks.
With n samples per test input of which c are correct, pass@1 is c/n and pass@2 is the unbiased estimate
1 - C(n-c, 2) / C(n, 2). Verbosity is the mean number of output tokens per rollout.

    python benchmarks/arc_agi_2/score.py --rows benchmarks/arc_agi_2/data/arc_agi_2_no_tools.jsonl --rollouts <rollouts.jsonl>
"""

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path


def output_tokens(rollout: dict) -> int:
    """Tokens the model generated in the rollout (arc_code_agent records them; other agents report usage)."""
    if rollout.get("generated_tokens") is not None:
        return int(rollout["generated_tokens"])
    usage = (rollout.get("response") or {}).get("usage") or {}
    return int(usage.get("output_tokens") or 0)


def pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        return float(c > 0)
    return 1.0 - comb(n - c, k) / comb(n, k)


def score(rows: list, rollouts: list) -> dict:
    by_row = defaultdict(list)
    tokens = []
    for r in rollouts:
        by_row[r["_ng_task_index"]].append(r.get("reward") == 1)
        tokens.append(output_tokens(r))
    tasks = defaultdict(lambda: {"pass@1": [], "pass@2": []})
    missing = 0
    for i, row in enumerate(rows):
        results = by_row.get(i, [])
        if not results:
            missing += 1
        n, c = len(results), sum(results)
        tasks[row["puzzle_id"]]["pass@1"].append(c / n if n else 0.0)
        tasks[row["puzzle_id"]]["pass@2"].append(pass_at_k(n, c, 2) if n else 0.0)
    task_scores = {k: {m: sum(v) / len(v) for m, v in s.items()} for k, s in tasks.items()}
    samples = sorted({len(v) for v in by_row.values()})
    return {
        "tasks": len(task_scores),
        "test_inputs": len(rows),
        "rollouts": len(rollouts),
        "samples_per_input": samples,
        "test_inputs_without_rollouts": missing,
        "pass@1": 100 * sum(t["pass@1"] for t in task_scores.values()) / len(task_scores),
        "pass@2": 100 * sum(t["pass@2"] for t in task_scores.values()) / len(task_scores),
        "mean_output_tokens": sum(tokens) / len(tokens) if tokens else None,
        "task_scores": task_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, required=True, help="the benchmark rows the rollouts were collected from")
    parser.add_argument("--rollouts", type=Path, required=True, help="Gym rollouts JSONL")
    parser.add_argument("--output", type=Path, help="write the full result (with per-task scores) here")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rows.open()]
    rollouts = [json.loads(line) for line in args.rollouts.open()]
    result = score(rows, rollouts)
    if args.output:
        args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "task_scores"}, indent=1))


if __name__ == "__main__":
    main()
