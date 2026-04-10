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
#!/usr/bin/env python3
"""Breakdown rollout metrics for format_verification - freeform formatting (ds2).

Usage:
  python misc/breakdown_freeform.py -f rollouts/ds2_freeform_formatting.jsonl
"""

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


PATTERN_CATEGORIES = {
    "bullet_asterisk": "bullets",
    "bullet_dash": "bullets",
    "bullet_arrow": "bullets",
    "bullet_double_dash": "bullets",
    "numbered_dot": "numbered",
    "numbered_letter": "numbered",
    "numbered_roman": "numbered",
    "numbered_paren": "numbered",
    "heading_hash": "headings",
    "heading_equals": "headings",
    "heading_underline": "headings",
    "heading_caps": "headings",
    "kv_colon": "key_value",
    "kv_arrow": "key_value",
    "kv_equals": "key_value",
    "kv_quoted": "key_value",
    "table_markdown": "tables",
    "table_pipe": "tables",
    "delimiter_hr": "delimiters",
    "delimiter_blank": "delimiters",
    "delimiter_equals": "delimiters",
    "delimiter_stars": "delimiters",
    "web_numbered_steps": "web",
    "web_direct_headers": "web",
    "web_mixed_paragraphs_bullets": "web",
}


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
    parser = argparse.ArgumentParser(description="Breakdown freeform formatting rollout metrics")
    parser.add_argument("-f", "--in-path", required=True)
    args = parser.parse_args()

    rows = list(iter_jsonl(args.in_path))
    if not rows:
        print("No rows found.")
        return

    w = max(60, len(args.in_path) + 4)
    print("=" * w)
    print("  Freeform Formatting (ds2) Breakdown")
    print(f"  {args.in_path}")
    print("=" * w)
    print()

    print_section("OVERALL", rows)

    print("-" * w)
    print("  By pattern_id")
    print("-" * w)
    print()

    by_pid = defaultdict(list)
    for r in rows:
        pid = r.get("verifier", {}).get("pattern_id", "unknown")
        by_pid[pid].append(r)
    for pid in sorted(by_pid):
        print_section(f"{pid}", by_pid[pid], indent=4)

    print("-" * w)
    print("  By pattern_category")
    print("-" * w)
    print()

    by_cat = defaultdict(list)
    for r in rows:
        pid = r.get("verifier", {}).get("pattern_id", "unknown")
        cat = PATTERN_CATEGORIES.get(pid, "other")
        by_cat[cat].append(r)
    for cat in sorted(by_cat):
        print_section(f"{cat}", by_cat[cat], indent=4)

    failures = [r for r in rows if r.get("reward", 0) != 1.0]
    if failures:
        print("-" * w)
        print(f"  Failure details ({len(failures)} failures)")
        print("-" * w)
        print()

        for r in failures:
            pid = r.get("verifier", {}).get("pattern_id", "?")
            md = r.get("match_details", {})
            ml = md.get("matching_lines", 0)
            mm = md.get("min_matches", 0)
            print(f"    {pid}: {ml} matching lines (need >= {mm}, gap={mm - ml})")
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
    summary_path = in_p.parent / f"{in_p.stem}_freeform_formatting_breakdown_summary.txt"
    summary_path.write_text(output)
    print(f"Summary written to {summary_path}")
