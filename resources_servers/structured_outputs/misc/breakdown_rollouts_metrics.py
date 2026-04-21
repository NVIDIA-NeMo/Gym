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
"""Breakdown rollout metrics for structured_outputs.

Unified script replacing both the old breakdown_rollouts_metrics.py and
breakdown_all_formats.py. Outputs clean tables with breakdowns by schema_type,
problem_type, schema_repr, error_type, and optional cross-tabs.

Usage:
  python misc/breakdown_rollouts_metrics.py -f rollouts/ds1.jsonl
  python misc/breakdown_rollouts_metrics.py -f rollouts/ds1.jsonl -v
"""

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


def iter_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pct(num, den):
    return f"{100 * num / den:.1f}%" if den else "  N/A"


def _group_by(rows, key, default="unknown"):
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(key, default)].append(r)
    return groups


def _stats(rows):
    n = len(rows)
    rewards = [r.get("reward", 0.0) for r in rows]
    n_pass = sum(1 for r in rewards if r == 1.0)
    return n, n_pass, mean(rewards) if rewards else 0.0


def print_table(title, groups, total_n):
    """Print a breakdown table for the given groups dict."""
    if not groups:
        return

    print(f"\n  {title}")
    header = f"  {'Category':<30} {'N':>6}  {'Pass':>6}  {'Rate':>7}  {'Mean':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for key in sorted(groups):
        n, n_pass, avg = _stats(groups[key])
        share = f"({pct(n, total_n).strip()})" if total_n else ""
        print(f"  {key:<30} {n:>6}  {n_pass:>6}  {pct(n_pass, n):>7}  {avg:>7.4f}  {share}")
    print()


def print_cross_tab(title, rows, row_key, col_key, row_default="unknown", col_default="unknown"):
    """Print a cross-tabulation of pass rates."""
    by_row = defaultdict(lambda: defaultdict(list))
    col_keys = set()
    for r in rows:
        rk = r.get(row_key, row_default)
        ck = r.get(col_key, col_default)
        by_row[rk][ck].append(r)
        col_keys.add(ck)

    cols = sorted(col_keys)
    if not cols or not by_row:
        return

    col_w = max(10, max(len(c) for c in cols) + 2)
    row_w = max(len(k) for k in by_row) + 2

    print(f"\n  {title}")
    header = f"  {'':>{row_w}}" + "".join(f"{c:>{col_w}}" for c in cols)
    print(header)
    print("  " + "-" * (row_w + col_w * len(cols)))

    for rk in sorted(by_row):
        cells = []
        for ck in cols:
            group = by_row[rk].get(ck, [])
            if group:
                n, n_pass, _ = _stats(group)
                cells.append(f"{n_pass}/{n}")
            else:
                cells.append("-")
        print(f"  {rk:>{row_w}}" + "".join(f"{c:>{col_w}}" for c in cells))
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

    total_n, total_pass, total_mean = _stats(rows)

    w = 80
    print("=" * w)
    print("  Structured Outputs - Rollout Metrics Breakdown")
    print(f"  {args.in_path}")
    print("=" * w)

    print(
        f"\n  OVERALL:  n={total_n}  pass={total_pass}/{total_n} ({pct(total_pass, total_n)})  mean_reward={total_mean:.4f}"
    )

    # --- By schema_type (always present) ---
    print_table("By schema_type (output format)", _group_by(rows, "schema_type"), total_n)

    # --- By problem_type (if augmented data) ---
    by_pt = _group_by(rows, "problem_type", default=None)
    by_pt.pop(None, None)
    if by_pt:
        print_table("By problem_type", by_pt, total_n)

    # --- By schema_repr (if augmented data) ---
    by_repr = _group_by(rows, "schema_repr", default=None)
    by_repr.pop(None, None)
    if by_repr:
        print_table("By schema_repr (input schema format)", by_repr, total_n)

    # --- By num_turns (if multistep) ---
    by_turns = _group_by(rows, "num_turns", default=None)
    by_turns.pop(None, None)
    if by_turns:
        print_table("By num_turns", {str(k): v for k, v in by_turns.items()}, total_n)

    # --- By source_format (if translation) ---
    by_src = _group_by(rows, "source_format", default=None)
    by_src.pop(None, None)
    if by_src:
        print_table("By source_format (translation input)", by_src, total_n)

    # --- By schema_fields_count (legacy data) ---
    by_fields = _group_by(rows, "schema_fields_count", default=None)
    by_fields.pop(None, None)
    if by_fields:
        print_table("By schema_fields_count", by_fields, total_n)

    # --- Error breakdown ---
    failures = [r for r in rows if r.get("reward", 0) != 1.0]
    if failures:
        print("-" * w)
        print(f"  Error breakdown ({len(failures)} failures)")
        print("-" * w)

        by_err = _group_by(failures, "error_type")
        print_table("By error_type", by_err, len(failures))

        print_cross_tab(
            "Pass rate: schema_type x error_type",
            failures,
            "schema_type",
            "error_type",
        )

        if by_pt:
            print_cross_tab(
                "Pass rate: problem_type x schema_type",
                rows,
                "problem_type",
                "schema_type",
                row_default="unknown",
            )

        if args.verbose:
            print("  Sample error messages:")
            seen = set()
            for r in failures:
                et = r.get("error_type", "unknown")
                em = r.get("error_message", "")
                key = (et, em[:80])
                if key not in seen:
                    seen.add(key)
                    print(f"    [{et}] {em[:150]}")
                if len(seen) >= 15:
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
    summary_path = in_p.parent / f"{in_p.stem}_breakdown_summary.txt"
    summary_path.write_text(output)
    print(f"Summary written to {summary_path}")
