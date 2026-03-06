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
import argparse
import json
from collections import Counter, defaultdict
from statistics import mean, median


SUBSET_LABELS = {
    "malformed_thinking": "malformed_thinking (</think> not generated)",
    "malformed_tool_call": "malformed_tool_call (tool call not parsed correctly)",
}


def iter_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pct(num, den):
    return f"{100 * num / den:.1f}%" if den else "N/A"


def fmt_count(num, den):
    return f"{num} ({pct(num, den)})"


def get_error_types(rows):
    types = set()
    for r in rows:
        et = r.get("response_error_type")
        if et is not None:
            types.add(et)
    return sorted(types)


def print_row(label, rows, all_error_types):
    n = len(rows)
    if n == 0:
        return
    n_clean = sum(1 for r in rows if r.get("response_error_type") is None)
    n_error = n - n_clean
    error_counts = Counter(r.get("response_error_type") for r in rows if r.get("response_error_type") is not None)

    print(f"{label}")
    print(f"  n = {n}")
    print(f"  correct:  {fmt_count(n_clean, n)}")
    print(f"  error:    {fmt_count(n_error, n)}")
    for et in all_error_types:
        c = error_counts.get(et, 0)
        print(f"    - {et}: {fmt_count(c, n)}")
    print()


def main(args):
    rows = list(iter_jsonl(args.in_path))
    if not rows:
        print("No rows found.")
        return

    by_input = defaultdict(list)
    for r in rows:
        by_input[r["input_problem_type"]].append(r)

    all_error_types = get_error_types(rows)

    w = max(60, len(args.in_path) + 4)
    print("=" * w)
    print(f"  {args.in_path}")
    print("=" * w)
    print()

    print_row("OVERALL", rows, all_error_types)

    print("-" * w)
    print()

    for ipt in sorted(by_input):
        label = SUBSET_LABELS.get(ipt, ipt)
        print_row(label, by_input[ipt], all_error_types)

    print("=" * w)
    print("  Trajectory depth (input prefix)")
    print("=" * w)
    print()

    subsets = [("OVERALL", rows)]
    for ipt in sorted(by_input):
        subsets.append((ipt, by_input[ipt]))
    print_depth_table(subsets, by_input)


def get_input_messages(row):
    inp = row.get("responses_create_params", {}).get("input", [])
    return inp if isinstance(inp, list) else []


def count_total_messages(row):
    return len(get_input_messages(row))


def count_tool_calls(row):
    return sum(1 for m in get_input_messages(row) if isinstance(m, dict) and m.get("type") == "function_call")


def count_reasoning_blocks(row):
    return sum(1 for m in get_input_messages(row) if isinstance(m, dict) and m.get("type") == "reasoning")


def stats_dict(values):
    if not values:
        return {"mean": "n/a", "med": "n/a", "min": "n/a", "max": "n/a"}
    return {
        "mean": f"{mean(values):.1f}",
        "med": f"{median(values):.1f}",
        "min": str(min(values)),
        "max": str(max(values)),
    }


def print_depth_table(subsets, rows_by_subset):
    metrics = [
        ("total_messages", count_total_messages),
        ("tool_calls", count_tool_calls),
        ("reasoning_blocks", count_reasoning_blocks),
    ]
    groups = ["all", "correct", "error"]
    stat_keys = ["mean", "med", "min", "max"]

    short_names = {"total_messages": "msgs", "tool_calls": "tools", "reasoning_blocks": "reason"}
    table_headers = ["subset", "group", "n"]
    for mname in [m[0] for m in metrics]:
        sn = short_names.get(mname, mname)
        for sk in stat_keys:
            table_headers.append(f"{sn}.{sk}")

    table_rows = []
    for subset_name, subset_rows in subsets:
        clean = [r for r in subset_rows if r.get("response_error_type") is None]
        error = [r for r in subset_rows if r.get("response_error_type") is not None]
        group_map = {"all": subset_rows, "correct": clean, "error": error}

        for gi, gname in enumerate(groups):
            group = group_map[gname]
            row = [subset_name if gi == 0 else "", gname, str(len(group))]
            for _, mfunc in metrics:
                sd = stats_dict([mfunc(r) for r in group] if group else [])
                for sk in stat_keys:
                    row.append(sd[sk])
            table_rows.append(row)
        table_rows.append(None)

    widths = [len(h) for h in table_headers]
    for row in table_rows:
        if row is None:
            continue
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells):
        parts = []
        for i, (cell, w) in enumerate(zip(cells, widths)):
            align = "<" if i <= 1 else ">"
            parts.append(f"{cell:{align}{w}}")
        return "  ".join(parts)

    metric_names = [m[0] for m in metrics]
    group_header_parts = [" " * widths[0], " " * widths[1], " " * widths[2]]
    for mname in metric_names:
        idx = metric_names.index(mname)
        span = sum(widths[3 + idx * 4 + j] for j in range(4)) + 2 * 3
        group_header_parts.append(f"{mname:^{span}}")
    print("  ".join(group_header_parts))

    print(fmt(table_headers))
    print("  ".join("-" * w for w in widths))
    for row in table_rows:
        if row is None:
            print()
        else:
            print(fmt(row))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--in-path", required=True)
    args = parser.parse_args()
    main(args)
