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
"""Breakdown rollout metrics for format_verification - citation format (ds3).

Usage:
  python misc/breakdown_citation.py -f rollouts/ds3_citation_format.jsonl
"""

import argparse
import io
import json
import re
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


def infer_reference_style(expected_markers):
    """Infer the reference style pattern from a list of expected markers."""
    if not expected_markers:
        return "unknown"
    m = expected_markers[0]
    if "source:" in m:
        return "[source:N]"
    if "web:" in m:
        return "[web:N]"
    if "ref:" in m:
        if m.startswith("["):
            return "[ref:N]"
        if m.startswith("<"):
            return "<ref:N>"
        if m.startswith("{"):
            return "{ref:N}"
    if "Part" in m:
        return "(Part N)"
    if m.startswith("(ref"):
        return "(ref N)"
    if m.startswith("<<"):
        return "<<N>>"
    if re.match(r"^\[\d+\]$", m):
        return "[N]"
    return "other"


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
    parser = argparse.ArgumentParser(description="Breakdown citation format rollout metrics")
    parser.add_argument("-f", "--in-path", required=True)
    args = parser.parse_args()

    rows = list(iter_jsonl(args.in_path))
    if not rows:
        print("No rows found.")
        return

    w = max(60, len(args.in_path) + 4)
    print("=" * w)
    print("  Citation Format (ds3) Breakdown")
    print(f"  {args.in_path}")
    print("=" * w)
    print()

    print_section("OVERALL", rows)

    print("-" * w)
    print("  By reference_style")
    print("-" * w)
    print()

    by_style = defaultdict(list)
    for r in rows:
        expected = r.get("match_details", {}).get("expected", [])
        style = infer_reference_style(expected)
        by_style[style].append(r)
    for style in sorted(by_style):
        print_section(f"{style}", by_style[style], indent=4)

    print("-" * w)
    print("  By ref_type (single vs multi)")
    print("-" * w)
    print()

    by_reftype = defaultdict(list)
    for r in rows:
        expected = r.get("match_details", {}).get("expected", [])
        rtype = "multi" if len(expected) > 1 else "single"
        by_reftype[rtype].append(r)
    for rt in sorted(by_reftype):
        print_section(f"{rt}", by_reftype[rt], indent=4)

    failures = [r for r in rows if r.get("reward", 0) != 1.0]
    if failures:
        print("-" * w)
        print(f"  Error breakdown ({len(failures)} failures)")
        print("-" * w)
        print()

        n_missing = sum(1 for r in failures if r.get("match_details", {}).get("missing"))
        n_spurious = sum(1 for r in failures if r.get("match_details", {}).get("spurious"))
        n_both = sum(
            1
            for r in failures
            if r.get("match_details", {}).get("missing") and r.get("match_details", {}).get("spurious")
        )
        print(f"    missing_markers:  {n_missing}")
        print(f"    spurious_markers: {n_spurious}")
        print(f"    both:             {n_both}")
        print()

        missing_counts = Counter()
        for r in failures:
            for m in r.get("match_details", {}).get("missing", []):
                style = infer_reference_style([m])
                missing_counts[style] += 1
        if missing_counts:
            print("    Missing markers by style:")
            for style, count in missing_counts.most_common():
                print(f"      {style}: {count}")
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
    summary_path = in_p.parent / f"{in_p.stem}_citation_format_breakdown_summary.txt"
    summary_path.write_text(output)
    print(f"Summary written to {summary_path}")
