#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Aggregate XSTest benchmark results from rollout JSONL.

Reads rollout output and computes:
- Overall score, unsafe rate, over-refusal rate
- Per-category breakdown (10 safe + 10 unsafe categories)
- Judge error/empty rate (when LLM judge is used)
- Verdict distribution

Usage:
    python aggregate_results.py --input results/xstest_rollouts.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def load_rollouts(fpath: str) -> list[dict]:
    with open(fpath, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def aggregate(entries: list[dict]) -> None:
    safe = [e for e in entries if e.get("label") == "safe"]
    unsafe = [e for e in entries if e.get("label") == "unsafe"]

    safe_correct = sum(1 for e in safe if e["reward"] == 1.0)
    unsafe_correct = sum(1 for e in unsafe if e["reward"] == 1.0)
    total_correct = safe_correct + unsafe_correct
    overall_rate = total_correct / len(entries)

    safe_refusals = sum(1 for e in safe if e["is_refusal"])
    unsafe_compliances = sum(1 for e in unsafe if not e["is_refusal"])

    # Judge statistics
    judge_errors = 0
    judge_empty = 0
    judge_total = 0
    verdict_counts: dict[str, int] = defaultdict(int)
    for e in entries:
        for judge_eval in e.get("judge_evaluations") or []:
            judge_total += 1
            vl = judge_eval.get("verdict_label")
            if vl == "judge_error":
                judge_errors += 1
            elif vl is None:
                judge_empty += 1
            else:
                verdict_counts[vl] += 1

    # Summary
    print("=" * 70)
    print("XSTest Benchmark Results")
    print("=" * 70)
    print()
    print(f"{'Metric':<35s} {'Value':>15s}")
    print("-" * 50)
    print(f"{'Overall score':<35s} {total_correct}/{len(entries)} ({overall_rate:.1%})")
    if unsafe:
        print(
            f"{'Unsafe rate (should_refuse failed)':<35s}"
            f" {unsafe_compliances}/{len(unsafe)} ({unsafe_compliances / len(unsafe):.1%})"
        )
    if safe:
        print(f"{'Over-refusal rate':<35s} {safe_refusals}/{len(safe)} ({safe_refusals / len(safe):.1%})")
    if judge_total > 0:
        print(f"{'Judge error rate':<35s} {judge_errors}/{judge_total} ({judge_errors / judge_total:.1%})")
        print(f"{'Judge empty rate':<35s} {judge_empty}/{judge_total} ({judge_empty / judge_total:.1%})")

    # Verdict distribution (if judge was used)
    if verdict_counts:
        print()
        print("Verdict distribution:")
        for verdict in sorted(verdict_counts.keys()):
            count = verdict_counts[verdict]
            print(f"  {verdict:<30s} {count:>5d} ({count / judge_total:.1%})")

    # Per-category breakdown (prompt_type is a top-level field on XSTestVerifyResponse)
    category_data: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        category_data[e.get("prompt_type", "unknown")].append(e)

    if len(category_data) > 1 or "unknown" not in category_data:
        print()
        print("=" * 70)
        print(f"{'Category':<40s} {'Label':>6s} {'Count':>6s} {'Pass Rate':>10s}")
        print("-" * 70)
        for cat in sorted(category_data.keys()):
            items = category_data[cat]
            label = items[0].get("label", "?")
            count = len(items)
            pass_rate = sum(1 for e in items if e["reward"] == 1.0) / count
            print(f"{cat:<40s} {label:>6s} {count:>6d} {pass_rate:>9.1%}")

    # Safe vs unsafe summary
    print()
    print("=" * 70)
    print(f"{'Group':<40s} {'Count':>6s} {'Pass Rate':>10s}")
    print("-" * 70)
    if safe:
        print(f"{'Safe prompts (should comply)':<40s} {len(safe):>6d} {safe_correct / len(safe):>9.1%}")
    if unsafe:
        print(f"{'Unsafe prompts (should refuse)':<40s} {len(unsafe):>6d} {unsafe_correct / len(unsafe):>9.1%}")
    print(f"{'OVERALL':<40s} {len(entries):>6d} {overall_rate:>9.1%}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="XSTest results aggregation")
    parser.add_argument("--input", required=True, help="Path to rollout JSONL")
    args = parser.parse_args()

    entries = load_rollouts(args.input)
    if not entries:
        print(f"No entries found in {args.input}", file=sys.stderr)
        sys.exit(1)

    aggregate(entries)


if __name__ == "__main__":
    main()
