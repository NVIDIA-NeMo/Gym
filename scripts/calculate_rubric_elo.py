#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compute ELO ratings from rubric score JSONL files for two models.

Takes two models' per-task rubric score files and derives ELO ratings by
comparing their average_score on each shared task.

Input JSONL format (one line per task):
    {"task_id": "xxx", "task_index": 0, "scores": [15.0, 14.5],
     "max_possible_scores": [20.0, 20.0], "average_score": 14.75,
     "overall_score_percentage": 73.75, "judgement_success": true}

Usage:
    python scripts/calculate_rubric_elo.py \\
        --model-a-scores-file output/gdpval/model_a_scores.jsonl \\
        --model-a-name "Qwen3-235B" \\
        --model-b-scores-file output/gdpval/model_b_scores.jsonl \\
        --model-b-name "Nemotron-Super-V3" \\
        --output-dir output/gdpval/rubric_elo
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime


def load_scores(filepath: str) -> dict[str, dict]:
    """Load a rubric score JSONL file into a dict keyed by task_id."""
    scores: dict[str, dict] = {}
    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping malformed JSON at {filepath}:{line_num}: {e}", file=sys.stderr)
                continue
            task_id = entry.get("task_id")
            if task_id is None:
                print(f"WARNING: Skipping entry without task_id at {filepath}:{line_num}", file=sys.stderr)
                continue
            scores[task_id] = entry
    return scores


def compare_tasks(
    scores_a: dict[str, dict], scores_b: dict[str, dict]
) -> tuple[list[dict], list[dict]]:
    """Compare two models on shared tasks by average_score.

    Returns (comparison_log, error_log).
    """
    shared_ids = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    comparison_log: list[dict] = []
    error_log: list[dict] = []

    for task_id in shared_ids:
        entry_a = scores_a[task_id]
        entry_b = scores_b[task_id]

        # Skip tasks where judgement failed for either model
        # Support both "judgement_success" and "success" field names
        success_a = entry_a.get("judgement_success", entry_a.get("success"))
        success_b = entry_b.get("judgement_success", entry_b.get("success"))
        if not success_a or not success_b:
            error_log.append({
                "task_id": task_id,
                "reason": "judgement failed for one or both models",
                "model_a_success": success_a,
                "model_b_success": success_b,
            })
            continue

        # Verify max_possible_scores match and are positive
        # Support both "max_possible_scores" (list) and "max_possible_score" (float)
        max_a = entry_a.get("max_possible_scores") or entry_a.get("max_possible_score")
        max_b = entry_b.get("max_possible_scores") or entry_b.get("max_possible_score")
        # Normalize to float for comparison
        max_a_val = max_a if isinstance(max_a, (int, float)) else (sum(max_a) if max_a else 0)
        max_b_val = max_b if isinstance(max_b, (int, float)) else (sum(max_b) if max_b else 0)
        if abs(max_a_val - max_b_val) > 0.01:
            error_log.append({
                "task_id": task_id,
                "reason": "max_possible_score mismatch",
                "model_a_max": max_a_val,
                "model_b_max": max_b_val,
            })
            continue
        if max_a_val <= 0:
            error_log.append({
                "task_id": task_id,
                "reason": "max_possible_score is non-positive",
                "max_possible_score": max_a_val,
            })
            continue

        score_a = entry_a.get("average_score", 0.0)
        score_b = entry_b.get("average_score", 0.0)

        if score_a > score_b:
            winner = "A"
        elif score_b > score_a:
            winner = "B"
        else:
            winner = "tie"

        comparison_log.append({
            "task_id": task_id,
            "model_a_score": score_a,
            "model_b_score": score_b,
            "max_possible_score": max_a_val,
            "winner": winner,
        })

    return comparison_log, error_log


def calculate_elo(score_rate: float, ref_elo: float) -> tuple[float, float]:
    """Compute ELO from a win/score rate against a reference.

    Returns (elo, normalized_elo) where normalized = (elo - 500) / 2000.
    """
    score_rate = max(0.001, min(0.999, score_rate))
    elo = ref_elo + 400.0 * math.log10(score_rate / (1.0 - score_rate))
    normalized_elo = (elo - 500.0) / 2000.0
    return elo, normalized_elo


def compute_elos(
    comparison_log: list[dict],
    ref_elo_a: int | None,
    ref_elo_b: int | None,
    name_a: str,
    name_b: str,
) -> dict:
    """Compute ELO predictions from comparison results."""
    wins_a = sum(1 for c in comparison_log if c["winner"] == "A")
    wins_b = sum(1 for c in comparison_log if c["winner"] == "B")
    ties = sum(1 for c in comparison_log if c["winner"] == "tie")
    total = len(comparison_log)

    if total == 0:
        return {
            "error": "No comparable tasks found",
            "model_a": name_a,
            "model_b": name_b,
        }

    # Score rates: wins + 0.5 * ties, divided by total
    score_rate_a = (wins_a + 0.5 * ties) / total
    score_rate_b = (wins_b + 0.5 * ties) / total

    result: dict = {
        "model_a": name_a,
        "model_b": name_b,
        "total_tasks": total,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_a": wins_a / total,
        "win_rate_b": wins_b / total,
        "tie_rate": ties / total,
        "score_rate_a": score_rate_a,
        "score_rate_b": score_rate_b,
    }

    # Symmetric anchor: both ref ELOs provided
    if ref_elo_a is not None and ref_elo_b is not None:
        anchor = (ref_elo_a + ref_elo_b) / 2.0
        elo_b_from_a, norm_b_from_a = calculate_elo(score_rate_b, ref_elo_a)
        elo_a_from_b, norm_a_from_b = calculate_elo(score_rate_a, ref_elo_b)
        delta = elo_b_from_a - ref_elo_a
        result["symmetric_anchor"] = anchor
        result["elo_a"] = anchor - delta / 2.0
        result["elo_b"] = anchor + delta / 2.0
        result["normalized_elo_a"] = (result["elo_a"] - 500.0) / 2000.0
        result["normalized_elo_b"] = (result["elo_b"] - 500.0) / 2000.0
    else:
        # One-sided: compute B's ELO from A's reference
        ref = ref_elo_a if ref_elo_a is not None else (ref_elo_b if ref_elo_b is not None else 1000)
        elo_b, norm_b = calculate_elo(score_rate_b, ref)
        result["ref_elo"] = ref
        result["elo_b"] = elo_b
        result["normalized_elo_b"] = norm_b
        # Derive A if B ref was given
        if ref_elo_b is not None and ref_elo_a is None:
            elo_a, norm_a = calculate_elo(score_rate_a, ref_elo_b)
            result["elo_a"] = elo_a
            result["normalized_elo_a"] = norm_a

    return result


def main():
    parser = argparse.ArgumentParser(description="Compute ELO ratings from rubric score JSONL files")
    parser.add_argument("--model-a-scores-file", type=str, required=True)
    parser.add_argument("--model-a-name", type=str, required=True)
    parser.add_argument("--model-a-ref-elo", type=int, default=None)
    parser.add_argument("--model-b-scores-file", type=str, required=True)
    parser.add_argument("--model-b-name", type=str, required=True)
    parser.add_argument("--model-b-ref-elo", type=int, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    scores_a = load_scores(args.model_a_scores_file)
    scores_b = load_scores(args.model_b_scores_file)
    print(f"Loaded {len(scores_a)} tasks for {args.model_a_name}")
    print(f"Loaded {len(scores_b)} tasks for {args.model_b_name}")

    comparison_log, error_log = compare_tasks(scores_a, scores_b)

    elo_result = compute_elos(
        comparison_log,
        ref_elo_a=args.model_a_ref_elo,
        ref_elo_b=args.model_b_ref_elo,
        name_a=args.model_a_name,
        name_b=args.model_b_name,
    )
    elo_result["generated_at"] = datetime.now().isoformat(timespec="seconds")

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "comparison_log.json"), "w", encoding="utf-8") as f:
        json.dump(comparison_log, f, indent=2)

    with open(os.path.join(args.output_dir, "error_log.json"), "w", encoding="utf-8") as f:
        json.dump(error_log, f, indent=2)

    with open(os.path.join(args.output_dir, "predicted_elos.json"), "w", encoding="utf-8") as f:
        json.dump(elo_result, f, indent=2)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"  {args.model_a_name} vs {args.model_b_name}")
    print(f"{'=' * 50}")
    print(f"  Comparable tasks:  {elo_result.get('total_tasks', 0)}")
    print(f"  Skipped (errors):  {len(error_log)}")
    print(f"  A wins: {elo_result.get('wins_a', 0)}  |  B wins: {elo_result.get('wins_b', 0)}  |  Ties: {elo_result.get('ties', 0)}")
    if "score_rate_a" in elo_result:
        print(f"  Score rate A: {elo_result['score_rate_a']:.4f}  |  Score rate B: {elo_result['score_rate_b']:.4f}")
    if "elo_a" in elo_result:
        print(f"  ELO {args.model_a_name}: {elo_result['elo_a']:.1f} (normalized: {elo_result.get('normalized_elo_a', 'N/A')})")
    if "elo_b" in elo_result:
        print(f"  ELO {args.model_b_name}: {elo_result['elo_b']:.1f} (normalized: {elo_result.get('normalized_elo_b', 'N/A')})")
    print(f"{'=' * 50}")
    print(f"  Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
