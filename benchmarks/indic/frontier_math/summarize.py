# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize Gym rollouts by language, counting repeated attempts within each problem."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from benchmarks.indic.frontier_math.normalization import normalized_grade


def summarize(path: Path) -> dict:
    """Return strict and normalized pass@1 estimates plus grading diagnostics."""
    tasks = defaultdict(lambda: defaultdict(list))
    normalized_tasks = defaultdict(lambda: defaultdict(list))
    statuses = defaultdict(Counter)
    recoveries = defaultdict(Counter)
    review = defaultdict(dict)
    pending = defaultdict(dict)
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {
                "language_code",
                "row_id",
                "reward",
                "grading_status",
                "judge_pass_stage",
                "human_evaluation_pending",
            } - row.keys()
            if missing:
                raise ValueError(f"Line {line_number}: missing {sorted(missing)}; pass a verified Gym rollout file")
            if row["reward"] not in (0.0, 1.0):
                raise ValueError(f"Line {line_number}: expected a binary reward")
            code, row_id = row["language_code"], row["row_id"]
            tasks[code][row_id].append(float(row["reward"]))
            normalized = normalized_grade(row)
            normalized_tasks[code][row_id].append(normalized.reward)
            if normalized.recovery_method:
                recoveries[code][normalized.recovery_method] += 1
            statuses[code][row["grading_status"]] += 1
            review[code][row_id] = row["judge_pass_stage"]
            pending[code][row_id] = row["human_evaluation_pending"]
    if not tasks:
        raise ValueError("No verified rollouts found")
    rates = {code: {row_id: mean(values) for row_id, values in problems.items()} for code, problems in tasks.items()}
    normalized_rates = {
        code: {row_id: mean(values) for row_id, values in problems.items()}
        for code, problems in normalized_tasks.items()
    }
    result = {}
    for code, per_problem in sorted(rates.items()):
        counts = [len(values) for values in tasks[code].values()]
        passed_review = [
            rate
            for row_id, rate in per_problem.items()
            if review[code][row_id] in {"first_judge_pass", "passed_after_correction", "english_source"}
        ]
        paired = set(per_problem) & rates.get("en", {}).keys()
        result[code] = {
            "problems": len(per_problem),
            "attempts": sum(counts),
            "min_attempts_per_problem": min(counts),
            "max_attempts_per_problem": max(counts),
            "pass_at_1": mean(per_problem.values()),
            "normalized_pass_at_1": mean(normalized_rates[code].values()),
            "normalized_recovered_attempts": sum(recoveries[code].values()),
            "normalization_methods": dict(recoveries[code]),
            "complete_12_problem_coverage": len(per_problem) == 12,
            "translation_review_passed_problems": len(passed_review),
            "pass_at_1_translation_review_passed": mean(passed_review) if passed_review else None,
            "human_review_pending_problems": sum(pending[code].values()),
            "grading_statuses": dict(statuses[code]),
            "paired_english_problems": len(paired),
            "paired_delta_vs_english": mean(per_problem[key] - rates["en"][key] for key in paired) if paired else None,
        }
    return {
        "benchmark": "Indic FrontierMath public sample",
        "protocol": "exact_boxed_text_no_tools",
        "languages": result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(summarize(args.rollouts), indent=2)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
