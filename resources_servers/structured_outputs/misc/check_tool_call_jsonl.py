# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static sanity checks for structured-output tool-call JSONL data."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "resources_servers/structured_outputs/data/structured_outputs_v4_tool_call.jsonl"
SCHEMA_VALUE_EXEMPT_KEYS = {"enum", "const", "default", "examples"}
DIST_KEYS = [
    "response_mode",
    "tool_choice",
    "parallel_tool_calls",
    "num_tools",
    "num_distractors",
    "has_distractors",
    "tool_schema_mode",
    "distractor_style",
    "tool_union_mode",
    "tool_name_style",
    "instruction_layout",
    "instruction_detail_level",
    "system_instruction_style",
]


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                yield row_idx, json.loads(line)
            except json.JSONDecodeError as exc:
                yield row_idx, {"__json_error__": f"{type(exc).__name__}: {exc}"}


def path_join(path: str, part: str) -> str:
    return f"{path}/{part}"


def walk_tool_schema(
    value: Any,
    *,
    path: str = "$",
    key: str | None = None,
    in_properties_map: bool = False,
    check_vllm_compat: bool = True,
):
    if key in SCHEMA_VALUE_EXEMPT_KEYS:
        return

    if isinstance(value, bool):
        if check_vllm_compat:
            yield "boolean_schema_node", path, value
        return

    if isinstance(value, list):
        for i, item in enumerate(value):
            yield from walk_tool_schema(
                item,
                path=path_join(path, str(i)),
                key=key,
                check_vllm_compat=check_vllm_compat,
            )
        return

    if not isinstance(value, dict):
        return

    if in_properties_map:
        for prop_name, prop_schema in value.items():
            prop_path = path_join(path, prop_name)
            if not isinstance(prop_schema, (dict, bool)):
                yield "invalid_property_schema", prop_path, prop_schema
                continue
            yield from walk_tool_schema(
                prop_schema,
                path=prop_path,
                key=prop_name,
                check_vllm_compat=check_vllm_compat,
            )
        return

    if check_vllm_compat and isinstance(value.get("format"), str):
        yield "format_annotation", path_join(path, "format"), value["format"]

    for child_key, child_value in value.items():
        yield from walk_tool_schema(
            child_value,
            path=path_join(path, child_key),
            key=child_key,
            in_properties_map=child_key == "properties" and isinstance(child_value, dict),
            check_vllm_compat=check_vllm_compat,
        )


def schema_contains_property_key(value: Any, property_key: str) -> bool:
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict) and property_key in properties:
            return True
        return any(schema_contains_property_key(child, property_key) for child in value.values())
    if isinstance(value, list):
        return any(schema_contains_property_key(child, property_key) for child in value)
    return False


def value_key(value: Any) -> str:
    if value is None:
        return "None"
    return str(value)


def add_issue(
    issues: list[dict[str, Any]],
    *,
    row_idx: int,
    severity: str,
    code: str,
    message: str,
    source_record_id: Any = None,
) -> None:
    issues.append(
        {
            "row_idx": row_idx,
            "source_record_id": source_record_id,
            "severity": severity,
            "code": code,
            "message": message,
        }
    )


def check_row(row_idx: int, row: dict[str, Any], *, require_response_mode: bool, check_vllm_compat: bool):
    issues: list[dict[str, Any]] = []
    source_record_id = row.get("source_record_id")

    if "__json_error__" in row:
        add_issue(
            issues,
            row_idx=row_idx,
            severity="error",
            code="json_parse_error",
            message=row["__json_error__"],
            source_record_id=source_record_id,
        )
        return issues

    if require_response_mode and row.get("response_mode") != "tool_call":
        add_issue(
            issues,
            row_idx=row_idx,
            severity="error",
            code="bad_response_mode",
            message=f"Expected response_mode='tool_call', got {row.get('response_mode')!r}",
            source_record_id=source_record_id,
        )

    responses_create_params = row.get("responses_create_params")
    if not isinstance(responses_create_params, dict):
        add_issue(
            issues,
            row_idx=row_idx,
            severity="error",
            code="missing_responses_create_params",
            message="responses_create_params must be an object",
            source_record_id=source_record_id,
        )
        return issues

    for mirrored_key in ("tool_choice", "parallel_tool_calls"):
        row_value = row.get(mirrored_key)
        params_value = responses_create_params.get(mirrored_key)
        if row_value is not None and params_value is not None and row_value != params_value:
            add_issue(
                issues,
                row_idx=row_idx,
                severity="error",
                code=f"{mirrored_key}_mismatch",
                message=f"{mirrored_key}={row_value!r}, responses_create_params.{mirrored_key}={params_value!r}",
                source_record_id=source_record_id,
            )

    tools = responses_create_params.get("tools")
    if not isinstance(tools, list) or not tools:
        add_issue(
            issues,
            row_idx=row_idx,
            severity="error",
            code="missing_tools",
            message="responses_create_params.tools must be a non-empty list",
            source_record_id=source_record_id,
        )
        return issues

    tool_by_name: dict[str, dict[str, Any]] = {}
    tool_names = []
    for tool_idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            add_issue(
                issues,
                row_idx=row_idx,
                severity="error",
                code="bad_tool",
                message=f"tools[{tool_idx}] must be an object",
                source_record_id=source_record_id,
            )
            continue
        if tool.get("type") != "function":
            add_issue(
                issues,
                row_idx=row_idx,
                severity="error",
                code="bad_tool_type",
                message=f"tools[{tool_idx}].type must be 'function', got {tool.get('type')!r}",
                source_record_id=source_record_id,
            )
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            add_issue(
                issues,
                row_idx=row_idx,
                severity="error",
                code="bad_tool_name",
                message=f"tools[{tool_idx}].name must be a non-empty string",
                source_record_id=source_record_id,
            )
            continue
        tool_names.append(name)
        if name in tool_by_name:
            add_issue(
                issues,
                row_idx=row_idx,
                severity="error",
                code="duplicate_tool_name",
                message=f"Duplicate tool name {name!r}",
                source_record_id=source_record_id,
            )
        tool_by_name[name] = tool

        parameters = tool.get("parameters")
        if not isinstance(parameters, dict):
            add_issue(
                issues,
                row_idx=row_idx,
                severity="error",
                code="bad_tool_parameters",
                message=f"tools[{tool_idx}].parameters must be an object",
                source_record_id=source_record_id,
            )
            continue
        for code, path, value in walk_tool_schema(parameters, check_vllm_compat=check_vllm_compat):
            add_issue(
                issues,
                row_idx=row_idx,
                severity="error",
                code=code,
                message=f"tools[{tool_idx}].parameters{path[1:]} = {value!r}",
                source_record_id=source_record_id,
            )

    expected_tool_name = row.get("tool_name")
    target_tool = None
    if expected_tool_name:
        target_tool = tool_by_name.get(expected_tool_name)
        if target_tool is None:
            add_issue(
                issues,
                row_idx=row_idx,
                severity="error",
                code="target_tool_missing",
                message=f"tool_name {expected_tool_name!r} not found in tools {tool_names}",
                source_record_id=source_record_id,
            )

    tool_payload_key = row.get("tool_payload_key")
    if tool_payload_key and target_tool is not None:
        parameters = target_tool.get("parameters")
        if isinstance(parameters, dict) and not schema_contains_property_key(parameters, str(tool_payload_key)):
            add_issue(
                issues,
                row_idx=row_idx,
                severity="error",
                code="payload_key_not_in_target_tool_schema",
                message=f"tool_payload_key {tool_payload_key!r} not found in target tool parameters",
                source_record_id=source_record_id,
            )

    schema_str = row.get("schema_str")
    if schema_str is not None:
        try:
            json.loads(schema_str)
        except json.JSONDecodeError as exc:
            add_issue(
                issues,
                row_idx=row_idx,
                severity="error",
                code="schema_str_parse_error",
                message=f"{type(exc).__name__}: {exc}",
                source_record_id=source_record_id,
            )

    num_tools = row.get("num_tools")
    if num_tools is not None and num_tools != len(tools):
        add_issue(
            issues,
            row_idx=row_idx,
            severity="error",
            code="num_tools_mismatch",
            message=f"num_tools={num_tools!r}, actual tools={len(tools)}",
            source_record_id=source_record_id,
        )

    num_distractors = row.get("num_distractors")
    tool_union_mode = row.get("tool_union_mode")
    distractor_style = row.get("distractor_style")
    if num_distractors == 0 and (distractor_style != "none" or tool_union_mode is not None or len(tools) != 1):
        add_issue(
            issues,
            row_idx=row_idx,
            severity="error",
            code="bad_no_distractor_shape",
            message="No-distractor row must use distractor_style='none', one tool, and no tool_union_mode",
            source_record_id=source_record_id,
        )
    if tool_union_mode is not None and (not num_distractors or len(tools) != 1 or not tool_payload_key):
        add_issue(
            issues,
            row_idx=row_idx,
            severity="error",
            code="bad_union_shape",
            message="Union rows must have distractors, one tool, and a target tool_payload_key",
            source_record_id=source_record_id,
        )

    return issues


def print_distribution(rows: list[dict[str, Any]]) -> None:
    print("\nDistributions:")
    for key in DIST_KEYS:
        counter = Counter(value_key(row.get(key)) for row in rows if key in row)
        if not counter:
            continue
        print(f"  {key}:")
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            print(f"    {value}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT)
    parser.add_argument("--max-errors", type=int, default=30)
    parser.add_argument("--require-response-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-vllm-compat", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for row_idx, row in iter_jsonl(input_path):
        rows.append(row)
        issues.extend(
            check_row(
                row_idx,
                row,
                require_response_mode=args.require_response_mode,
                check_vllm_compat=args.check_vllm_compat,
            )
        )

    error_counts = Counter(issue["code"] for issue in issues if issue["severity"] == "error")
    print(f"Checked {len(rows)} rows from {input_path}")
    if error_counts:
        print("\nErrors:")
        for code, count in sorted(error_counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {code}: {count}")
    else:
        print("\nErrors: 0")

    print_distribution(rows)

    if issues:
        print(f"\nFirst {min(args.max_errors, len(issues))} issues:")
        for issue in issues[: args.max_errors]:
            print(
                f"  row={issue['row_idx']} source_record_id={issue.get('source_record_id')} "
                f"{issue['severity']} {issue['code']}: {issue['message']}"
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
