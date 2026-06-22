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
    "tool_choice",
    "parallel_tool_calls",
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
USE_FIELD_DEFAULT: Any = object()


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


def default_for_key(key: str) -> Any:
    return None if key == "error_type" else "unknown"


def field_value_to_key(row: dict[str, Any], key: str, default: Any = USE_FIELD_DEFAULT) -> str:
    if default is USE_FIELD_DEFAULT:
        default = default_for_key(key)
    return value_to_key(row.get(key, default))


def failure_error_to_key(row: dict[str, Any]) -> str:
    error_type = row.get("error_type") or row.get("failure_signature") or row.get("verify_error_type")
    if error_type is None:
        return "unclassified"
    return value_to_key(error_type)


def category_sort_key(value: str) -> tuple[int, int | str, str]:
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return (0, int(value), value)
    return (1, value.casefold(), value)


def group_by(
    rows: list[dict[str, Any]], key: str, default: Any = USE_FIELD_DEFAULT
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[field_value_to_key(row, key, default=default)].append(row)
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


def observed_values(rows: list[dict[str, Any]], key: str, default: Any = USE_FIELD_DEFAULT) -> set[str]:
    return {field_value_to_key(row, key, default=default) for row in rows}


def has_multiple_values(rows: list[dict[str, Any]], key: str, default: Any = USE_FIELD_DEFAULT) -> bool:
    return len(observed_values(rows, key, default=default)) > 1


def print_table(
    title: str,
    groups: dict[str, list[dict[str, Any]]],
    total_n: int,
    *,
    skip_singleton: bool = False,
) -> bool:
    if not groups:
        return False
    if skip_singleton and len(groups) <= 1:
        return False

    print(f"\n  {title}")
    header = f"  {'Category':<34} {'N':>7}  {'Share':>7}  {'Pass':>7}  {'Rate':>7}  {'Mean':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for key, grouped_rows in sorted(groups.items(), key=lambda kv: category_sort_key(kv[0])):
        n, n_pass, avg = stats(grouped_rows)
        print(f"  {key:<34} {n:>7}  {pct(n, total_n):>7}  {n_pass:>7}  {pct(n_pass, n):>7}  {avg:>7.4f}")
    return True


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
        if "distractor_style" not in row and "tool_schema_mode" not in row:
            continue

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
        if distractor_style == "single_tool_multi_key":
            if not num_distractors or num_tools != 1 or not tool_payload_key or tool_union_mode is not None:
                errors.append(f"row {i}: bad single-tool multi-key shape")
    return errors


def print_optional_tables(
    rows: list[dict[str, Any]],
    total_n: int,
    fields: list[tuple[str, str]],
    *,
    skip_singletons: bool = True,
) -> None:
    for title, key in fields:
        print_table(title, optional_group_by(rows, key), total_n, skip_singleton=skip_singletons)


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
        ("By tool_choice", "tool_choice"),
        ("By parallel_tool_calls", "parallel_tool_calls"),
        ("By tool_schema_mode", "tool_schema_mode"),
        ("By distractor_style", "distractor_style"),
        ("By tool_union_mode", "tool_union_mode"),
        ("By num_tools", "num_tools"),
        ("By num_distractors", "num_distractors"),
        ("By has_distractors", "has_distractors"),
        ("By tool_name_style", "tool_name_style"),
        ("By instruction_layout", "instruction_layout"),
    ]:
        print_table(title, group_by(rows, key), total_n, skip_singleton=True)

    print_optional_tables(
        rows,
        total_n,
        [
            ("By instruction_detail_level", "instruction_detail_level"),
            ("By system_instruction_style", "system_instruction_style"),
        ],
    )


def print_failure_error_table(failures: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in failures:
        groups[failure_error_to_key(row)].append(row)

    if not groups:
        return

    print("\n  Failures by error_type")
    header = f"  {'Error Type':<34} {'Failures':>9}  {'Share':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    total_failures = len(failures)
    for key, grouped_rows in sorted(groups.items(), key=lambda kv: category_sort_key(kv[0])):
        n = len(grouped_rows)
        print(f"  {key:<34} {n:>9}  {pct(n, total_failures):>7}")


def print_failure_count_table(
    title: str, rows: list[dict[str, Any]], failures: list[dict[str, Any]], key: str
) -> bool:
    if not failures or not any(key in row for row in rows) or not has_multiple_values(rows, key):
        return False

    rows_by_key = group_by(rows, key)
    failures_by_key: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in failures:
        row_key = field_value_to_key(row, key)
        error_type = failure_error_to_key(row)
        failures_by_key[row_key][error_type] += 1

    category_width = max(18, max(len(row_key) for row_key in rows_by_key) + 2)
    top_width = 54
    header = f"  {'Category':<{category_width}}{'Rows':>8}  {'Pass':>8}  {'Rate':>7}  {'Failures':>9}  {'Fail Rate':>9}  {'Top failures':<{top_width}}"

    print(f"\n  {title}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for row_key, grouped_rows in sorted(rows_by_key.items(), key=lambda kv: category_sort_key(kv[0])):
        error_counts = failures_by_key.get(row_key, {})
        n_rows = len(grouped_rows)
        n_failures = sum(error_counts.values())
        _n, n_pass, _avg = stats(grouped_rows)
        top_failures = ", ".join(
            f"{error_type}={count}"
            for error_type, count in sorted(
                error_counts.items(), key=lambda item: (-item[1], category_sort_key(item[0]))
            )[:3]
        )
        top_failures = top_failures or "-"
        print(
            f"  {row_key:<{category_width}}"
            f"{n_rows:>8}  {n_pass:>8}  {pct(n_pass, n_rows):>7}  {n_failures:>9}  {pct(n_failures, n_rows):>9}  "
            f"{top_failures:<{top_width}}"
        )
    return True


def print_failure_breakdowns(
    rows: list[dict[str, Any]],
    verbose: bool,
    breakdown_keys: list[str],
    *,
    include_error_table: bool = True,
) -> None:
    failures = [row for row in rows if row.get("reward", 0.0) != 1.0]
    if not failures:
        return

    printable_breakdown_keys = [
        key for key in breakdown_keys if any(key in row for row in rows) and has_multiple_values(rows, key)
    ]
    if not include_error_table and not printable_breakdown_keys and not verbose:
        return

    print("\n" + "-" * 90)
    print(f"  Error breakdown ({len(failures)} failures)")
    print("-" * 90)
    if include_error_table:
        print_failure_error_table(failures)

    for key in printable_breakdown_keys:
        print_failure_count_table(f"Failures by {key}", rows, failures, key)

    if not verbose:
        return

    print("\n  Sample failures")
    seen = set()
    for row in failures:
        error_type = failure_error_to_key(row)
        error_message = value_to_key(row.get("error_message", ""))[:200]
        key = (error_type, error_message[:80])
        if key in seen:
            continue
        seen.add(key)
        print(f"    [{error_type}] {error_message}")
        if len(seen) >= 20:
            break


def v4_error_breakdown_keys() -> list[str]:
    return [
        "distractor_style",
        "tool_schema_mode",
        "tool_choice",
        "parallel_tool_calls",
        "num_tools",
        "num_distractors",
    ]


def legacy_error_breakdown_keys() -> list[str]:
    return [
        "schema_type",
        "source_schema_type",
        "problem_type",
    ]


def print_section_header(title: str, rows: list[dict[str, Any]]) -> None:
    n, n_pass, avg = stats(rows)
    print("\n" + "#" * 90)
    print(f"  {title}: n={n} pass={n_pass}/{n} ({pct(n_pass, n)}) mean_reward={avg:.4f}")
    print("#" * 90)


def render_report(in_path: Path, rows: list[dict[str, Any]], verbose: bool) -> str:
    total_n, total_pass, total_mean = stats(rows)
    rows_by_version = group_by_version(rows)
    multiple_versions = len(rows_by_version) > 1

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        print("=" * 90)
        print("  Structured Outputs - Rollout Metrics")
        print(f"  {in_path}")
        if multiple_versions:
            detected_versions = ", ".join(
                f"{version}={len(grouped_rows)}"
                for version, grouped_rows in sorted(rows_by_version.items(), key=lambda kv: category_sort_key(kv[0]))
            )
            print(f"  detected_versions={detected_versions}")
        print("=" * 90)
        print(
            f"\n  OVERALL: n={total_n} pass={total_pass}/{total_n} ({pct(total_pass, total_n)}) "
            f"mean_reward={total_mean:.4f}"
        )

        print_table("By detected_version", rows_by_version, total_n, skip_singleton=True)

        v4_rows = rows_by_version.get("v4_tool_call", [])
        if v4_rows:
            if multiple_versions:
                print_section_header("V4 tool-call rows", v4_rows)
            print_v4_breakdowns(v4_rows, len(v4_rows))
            print_failure_breakdowns(
                v4_rows,
                verbose,
                v4_error_breakdown_keys(),
            )

        legacy_rows = rows_by_version.get("legacy_text_output", [])
        if legacy_rows:
            if multiple_versions:
                print_section_header("Legacy text-output rows", legacy_rows)
            print_legacy_breakdowns(legacy_rows, len(legacy_rows))
            print_failure_breakdowns(legacy_rows, verbose, legacy_error_breakdown_keys())

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
