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
#!/usr/bin/env python3
"""Breakdown rollout metrics for structured_outputs (all formats).

Usage:
  python misc/breakdown_all_formats.py -f rollouts/ds1_all_formats.jsonl
  python misc/breakdown_all_formats.py -f rollouts/ds1_all_formats.jsonl -v
"""

import argparse
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def iter_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pct(num, den):
    return f"{100 * num / den:.1f}%" if den else "N/A"


def print_section(label, rows, indent=2):
    n = len(rows)
    if n == 0:
        return
    rewards = [r.get("reward", 0.0) for r in rows]
    n_pass = sum(1 for r in rewards if r == 1.0)
    prefix = " " * indent
    print(f"{prefix}{label}")
    print(f"{prefix}  n={n}  pass={n_pass}/{n} ({pct(n_pass, n)})  mean_reward={mean(rewards):.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Breakdown structured_outputs rollout metrics")
    parser.add_argument("-f", "--in-path", required=True)
    parser.add_argument("-v", "--verbose", action="store_true", help="Show sample error messages")
    args = parser.parse_args()

    rows = list(iter_jsonl(args.in_path))
    if not rows:
        print("No rows found.")
        return

    w = max(60, len(args.in_path) + 4)
    print("=" * w)
    print("  Structured Outputs - All Formats Breakdown")
    print(f"  {args.in_path}")
    print("=" * w)
    print()

    print_section("OVERALL", rows)

    print("-" * w)
    print("  By schema_type")
    print("-" * w)
    print()

    by_type = defaultdict(list)
    for r in rows:
        by_type[r.get("schema_type", "unknown")].append(r)
    for st in sorted(by_type):
        print_section(f"schema_type={st}", by_type[st], indent=4)

    failures = [r for r in rows if r.get("reward", 0) != 1.0]
    if failures:
        print("-" * w)
        print(f"  Error breakdown ({len(failures)} failures)")
        print("-" * w)
        print()

        error_counts = Counter()
        error_by_type = defaultdict(lambda: Counter())
        for r in failures:
            et = r.get("error_type", "unknown")
            st = r.get("schema_type", "unknown")
            error_counts[et] += 1
            error_by_type[st][et] += 1

        for et, count in error_counts.most_common():
            print(f"    {et}: {count}")
        print()

        print("    By schema_type x error_type:")
        for st in sorted(error_by_type):
            parts = ", ".join(f"{et}={c}" for et, c in error_by_type[st].most_common())
            print(f"      {st}: {parts}")
        print()

        if args.verbose:
            print("    Sample error messages:")
            seen = set()
            for r in failures:
                et = r.get("error_type", "unknown")
                em = r.get("error_message", "")
                key = (et, em[:80])
                if key not in seen:
                    seen.add(key)
                    print(f"      [{et}] {em[:120]}")
                if len(seen) >= 10:
                    break
            print()

    print("=" * w)


if __name__ == "__main__":
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    main()
    sys.stdout = old_stdout
    output = buf.getvalue()
    print(output, end="")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-f", "--in-path", required=True)
    args, _ = parser.parse_known_args()
    in_p = Path(args.in_path)
    summary_path = in_p.parent / f"{in_p.stem}_schema_adherence_breakdown_summary.txt"
    summary_path.write_text(output)
    print(f"Summary written to {summary_path}")
