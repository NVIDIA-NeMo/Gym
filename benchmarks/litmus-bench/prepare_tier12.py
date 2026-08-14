# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the bundled paired Litmus Tier 1/2 example benchmark."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
SOURCE_FPATH = DATA_DIR / "examples" / "litmus-tier12-paired_example.jsonl"
OUTPUT_FPATH = DATA_DIR / "litmus-tier12-paired_benchmark.jsonl"

EXPECTED_QUESTIONS_PER_TIER = 5
EXPECTED_METHODS = frozenset({"direct", "mcp-python"})
TOOL_NAME = "stateful_python_code_exec"
REQUIRED_FIELDS = frozenset(
    {
        "responses_create_params",
        "expected_answer",
        "answer_type",
        "property",
        "method",
        "uuid",
        "question_uuid",
        "pair_id",
        "tier",
        "tool_use",
    }
)


def _row_error(row_number: int, message: str) -> ValueError:
    return ValueError(f"Litmus Tier 1/2 example row {row_number}: {message}")


def _validate_row(row: Mapping[str, Any], row_number: int) -> None:
    missing = sorted(REQUIRED_FIELDS - row.keys())
    if missing:
        raise _row_error(row_number, f"missing required fields: {missing}")

    if row["tier"] not in {1, 2}:
        raise _row_error(row_number, f"tier must be 1 or 2, got {row['tier']!r}")
    if row["method"] not in EXPECTED_METHODS:
        raise _row_error(row_number, f"unsupported method {row['method']!r}")
    if row["tool_use"] is not (row["method"] == "mcp-python"):
        raise _row_error(row_number, "tool_use must agree with method")

    for field in ("uuid", "question_uuid", "pair_id", "property", "answer_type"):
        if not isinstance(row[field], str) or not row[field]:
            raise _row_error(row_number, f"{field} must be a non-empty string")
    if row["pair_id"] != row["question_uuid"]:
        raise _row_error(row_number, "pair_id must equal question_uuid")

    create_params = row["responses_create_params"]
    if not isinstance(create_params, Mapping):
        raise _row_error(row_number, "responses_create_params must be an object")
    messages = create_params.get("input")
    if not isinstance(messages, list) or not messages:
        raise _row_error(row_number, "responses_create_params.input must be a non-empty list")

    tools = create_params.get("tools", [])
    if not isinstance(tools, list):
        raise _row_error(row_number, "responses_create_params.tools must be a list when present")
    if row["method"] == "direct":
        if tools:
            raise _row_error(row_number, "direct row must not expose tools")
        return

    if len(tools) != 1 or tools[0].get("type") != "function" or tools[0].get("name") != TOOL_NAME:
        raise _row_error(row_number, f"mcp-python row must expose exactly the {TOOL_NAME!r} function")


def _render_rows(source_path: Path) -> str:
    rows: list[dict[str, Any]] = []
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with source_path.open(encoding="utf-8") as source:
        for row_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise _row_error(row_number, "row must be an object")
            row = dict(row)
            _validate_row(row, row_number)
            row.pop("agent_ref", None)
            rows.append(row)
            pairs[row["pair_id"]].append(row)

    tier_counts: Counter[int] = Counter()
    for pair_id, pair_rows in pairs.items():
        methods = {row["method"] for row in pair_rows}
        if len(pair_rows) != 2 or methods != EXPECTED_METHODS:
            raise ValueError(f"Litmus Tier 1/2 pair {pair_id!r} must contain one row per method")

        direct = next(row for row in pair_rows if row["method"] == "direct")
        tool = next(row for row in pair_rows if row["method"] == "mcp-python")
        for field in ("tier", "property", "expected_answer", "answer_type", "uuid", "question_uuid"):
            if direct[field] != tool[field]:
                raise ValueError(f"Litmus Tier 1/2 pair {pair_id!r} disagrees on {field}")
        if direct["responses_create_params"]["input"] != tool["responses_create_params"]["input"]:
            raise ValueError(f"Litmus Tier 1/2 pair {pair_id!r} does not use an identical prompt")
        tier_counts[direct["tier"]] += 1

    expected_counts = Counter({1: EXPECTED_QUESTIONS_PER_TIER, 2: EXPECTED_QUESTIONS_PER_TIER})
    if tier_counts != expected_counts:
        raise ValueError(
            f"Litmus Tier 1/2 example question counts are {dict(tier_counts)}; expected {dict(expected_counts)}"
        )

    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)


def _atomic_write(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(dir=output_path.parent, prefix=f".{output_path.name}.")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def prepare() -> Path:
    """Validate and materialize the bundled paired example."""
    content = _render_rows(SOURCE_FPATH)
    _atomic_write(OUTPUT_FPATH, content)
    print(f"Wrote {len(content.splitlines())} paired Litmus Tier 1/2 rows to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
