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
"""Break down rollout metrics for all structured_outputs data versions."""

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


V4_KEYS = {
    "response_mode",
    "tool_schema_mode",
    "distractor_style",
    "tool_union_mode",
    "num_distractors",
    "has_distractors",
    "tool_name_style",
    "instruction_layout",
    "instruction_detail_level",
    "system_instruction_style",
}


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pct(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}%" if den else "N/A"


def value_to_key(value: Any) -> str:
    if value is None:
        return "None"
    return str(value)


def group_by(rows: list[dict[str, Any]], key: str, default: Any = "unknown") -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[value_to_key(row.get(key, default))].append(row)
    return groups


def optional_group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups = group_by(rows, key, default=None)
    groups.pop("None", None)
    return groups


def stats(rows: list[dict[str, Any]]) -> tuple[int, int, float]:
    rewards = [float(row.get("reward", 0.0) or 0.0) for row in rows]
    n = len(rows)
    n_pass = sum(1 for reward in rewards if reward == 1.0)
    return n, n_pass, mean(rewards) if rewards else 0.0


def print_table(title: str, groups: dict[str, list[dict[str, Any]]], total_n: int) -> None:
    if not groups:
        return

    print(f"\n  {title}")
    header = f"  {'Category':<34} {'N':>7}  {'Share':>7}  {'Pass':>7}  {'Rate':>7}  {'Mean':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for key, grouped_rows in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        n, n_pass, avg = stats(grouped_rows)
        print(f"  {key:<34} {n:>7}  {pct(n, total_n):>7}  {n_pass:>7}  {pct(n_pass, n):>7}  {avg:>7.4f}")


def print_cross_tab(title: str, rows: list[dict[str, Any]], row_key: str, col_key: str) -> None:
    by_row: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    col_keys = set()
    for row in rows:
        row_value = value_to_key(row.get(row_key, "unknown"))
        col_value = value_to_key(row.get(col_key, "unknown"))
        by_row[row_value][col_value].append(row)
        col_keys.add(col_value)

    if not by_row or not col_keys:
        return

    cols = sorted(col_keys)
    row_width = max(18, max(len(row_value) for row_value in by_row) + 2)
    col_width = max(12, max(len(col) for col in cols) + 2)

    print(f"\n  {title}")
    print(f"  {'':>{row_width}}" + "".join(f"{col:>{col_width}}" for col in cols))
    print("  " + "-" * (row_width + col_width * len(cols)))

    for row_value in sorted(by_row):
        cells = []
        for col in cols:
            cell_rows = by_row[row_value].get(col, [])
            if not cell_rows:
                cells.append("-")
                continue
            n, n_pass, _ = stats(cell_rows)
            cells.append(f"{n_pass}/{n}")
        print(f"  {row_value:>{row_width}}" + "".join(f"{cell:>{col_width}}" for cell in cells))


def is_v4_row(row: dict[str, Any]) -> bool:
    return row.get("response_mode") == "tool_call" or bool(V4_KEYS & row.keys())


def row_version(row: dict[str, Any]) -> str:
    return "v4_tool_call" if is_v4_row(row) else "legacy_text_output"


def group_by_version(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row_version(row)].append(row)
    return groups


def validate_v4_invariants(rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    for i, row in enumerate(rows):
        num_distractors = row.get("num_distractors")
        tool_union_mode = row.get("tool_union_mode")
        num_tools = row.get("num_tools")
        distractor_style = row.get("distractor_style")
        tool_payload_key = row.get("tool_payload_key")

        if num_distractors == 0:
            if distractor_style != "none" or tool_union_mode is not None or num_tools != 1:
                errors.append(f"row {i}: bad no-distractor shape")
        if tool_union_mode is not None:
            if not num_distractors or num_tools != 1 or not tool_payload_key:
                errors.append(f"row {i}: bad union shape")
    return errors


def print_optional_tables(rows: list[dict[str, Any]], total_n: int, fields: list[tuple[str, str]]) -> None:
    for title, key in fields:
        print_table(title, optional_group_by(rows, key), total_n)


def print_legacy_breakdowns(rows: list[dict[str, Any]], total_n: int) -> None:
    print_optional_tables(
        rows,
        total_n,
        [
            ("By schema_type", "schema_type"),
            ("By problem_type", "problem_type"),
            ("By schema_repr", "schema_repr"),
            ("By num_turns", "num_turns"),
            ("By source_format", "source_format"),
            ("By schema_fields_count", "schema_fields_count"),
        ],
    )


def print_v4_breakdowns(rows: list[dict[str, Any]], total_n: int) -> None:
    invariant_errors = validate_v4_invariants(rows)
    print(f"\n  V4 invariant check: {'PASS' if not invariant_errors else 'FAIL'}")
    if invariant_errors:
        for error in invariant_errors[:20]:
            print(f"    {error}")
        if len(invariant_errors) > 20:
            print(f"    ... {len(invariant_errors) - 20} more")

    for title, key in [
        ("By response_mode", "response_mode"),
        ("By tool_schema_mode", "tool_schema_mode"),
        ("By distractor_style", "distractor_style"),
        ("By tool_union_mode", "tool_union_mode"),
        ("By num_distractors", "num_distractors"),
        ("By has_distractors", "has_distractors"),
        ("By tool_name_style", "tool_name_style"),
        ("By instruction_layout", "instruction_layout"),
        ("By error_type", "error_type"),
    ]:
        print_table(title, group_by(rows, key), total_n)

    print_optional_tables(
        rows,
        total_n,
        [
            ("By instruction_detail_level", "instruction_detail_level"),
            ("By system_instruction_style", "system_instruction_style"),
        ],
    )

    print_cross_tab("Pass counts: distractor_style x tool_schema_mode", rows, "distractor_style", "tool_schema_mode")
    print_cross_tab("Pass counts: num_distractors x error_type", rows, "num_distractors", "error_type")


def print_failure_breakdowns(rows: list[dict[str, Any]], verbose: bool, include_source_schema_type: bool = False) -> None:
    failures = [row for row in rows if row.get("reward", 0.0) != 1.0]
    if not failures:
        return

    print("\n" + "-" * 90)
    print(f"  Error breakdown ({len(failures)} failures)")
    print("-" * 90)
    print_table("By error_type", group_by(failures, "error_type"), len(failures))

    row_col_keys = [
        ("schema_type", "error_type"),
        ("problem_type", "schema_type"),
        ("distractor_style", "error_type"),
        ("tool_schema_mode", "error_type"),
    ]
    if include_source_schema_type:
        row_col_keys.insert(1, ("source_schema_type", "error_type"))

    for row_key, col_key in row_col_keys:
        if any(row_key in row and col_key in row for row in rows):
            print_cross_tab(f"Pass counts: {row_key} x {col_key}", rows, row_key, col_key)

    if not verbose:
        return

    print("\n  Sample failures")
    seen = set()
    for row in failures:
        error_type = value_to_key(row.get("error_type", "unknown"))
        error_message = value_to_key(row.get("error_message", ""))[:200]
        key = (error_type, error_message[:80])
        if key in seen:
            continue
        seen.add(key)
        print(f"    [{error_type}] {error_message}")
        if len(seen) >= 20:
            break


def print_section_header(title: str, rows: list[dict[str, Any]]) -> None:
    n, n_pass, avg = stats(rows)
    print("\n" + "#" * 90)
    print(f"  {title}: n={n} pass={n_pass}/{n} ({pct(n_pass, n)}) mean_reward={avg:.4f}")
    print("#" * 90)


def render_report(in_path: Path, rows: list[dict[str, Any]], verbose: bool) -> str:
    total_n, total_pass, total_mean = stats(rows)
    rows_by_version = group_by_version(rows)

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        print("=" * 90)
        print("  Structured Outputs - Rollout Metrics")
        print(f"  {in_path}")
        detected_versions = ", ".join(
            f"{version}={len(grouped_rows)}" for version, grouped_rows in rows_by_version.items()
        )
        print(f"  detected_versions={detected_versions}")
        print("=" * 90)
        print(
            f"\n  OVERALL: n={total_n} pass={total_pass}/{total_n} ({pct(total_pass, total_n)}) "
            f"mean_reward={total_mean:.4f}"
        )

        print_table("By detected_version", rows_by_version, total_n)

        v4_rows = rows_by_version.get("v4_tool_call", [])
        if v4_rows:
            print_section_header("V4 tool-call rows", v4_rows)
            print_v4_breakdowns(v4_rows, len(v4_rows))
            print_failure_breakdowns(v4_rows, verbose)

        legacy_rows = rows_by_version.get("legacy_text_output", [])
        if legacy_rows:
            print_section_header("Legacy text-output rows", legacy_rows)
            print_legacy_breakdowns(legacy_rows, len(legacy_rows))
            print_failure_breakdowns(legacy_rows, verbose, include_source_schema_type=True)

        if len(rows_by_version) > 1:
            print_section_header("All rows combined error view", rows)
            print_failure_breakdowns(rows, verbose)
        print("=" * 90)
    finally:
        sys.stdout = old_stdout
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Break down structured_outputs rollout metrics")
    parser.add_argument("-f", "--in-path", required=True)
    parser.add_argument("-v", "--verbose", action="store_true", help="Show sample error messages")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    rows = list(iter_jsonl(in_path))
    if not rows:
        output = "No rows found.\n"
    else:
        output = render_report(in_path, rows, args.verbose)

    print(output, end="")
    summary_path = in_path.parent / f"{in_path.stem}_breakdown_summary.txt"
    summary_path.write_text(output, encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
