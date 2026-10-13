# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the pinned Litmus-Bench v0.1 test split for NeMo Gym."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "litmus-bench_benchmark.jsonl"

HF_DATASET = "nvidia/Nemotron-RL-litmus-bench-v0.1"
HF_REVISION = "dab852b1fd2085f60c49af85fcb7d58ead4bf1c1"  # pragma: allowlist secret
HF_SPLIT = "test"
EXPECTED_TEST_ROWS = 482

SUPPORTED_PROPERTY_TYPES = frozenset({"bool", "count", "fragment", "presence"})
BOOLEAN_PROPERTY_TYPES = frozenset({"bool", "presence"})
REQUIRED_FIELDS = frozenset(
    {
        "license",
        "responses_create_params",
        "expected_answer",
        "property_type",
        "property",
        "chembl_id",
        "smiles",
        "method",
        "use_box_format",
        "messages",
        "metadata",
        "uuid",
        "used_in",
    }
)


def _row_error(row_number: int, message: str) -> ValueError:
    return ValueError(f"Litmus-Bench test row {row_number}: {message}")


def _validate_row(row: Mapping[str, Any], row_number: int) -> None:
    missing = sorted(REQUIRED_FIELDS - row.keys())
    if missing:
        raise _row_error(row_number, f"missing required fields: {missing}")

    if row["method"] != "direct":
        raise _row_error(row_number, f"method must be 'direct', got {row['method']!r}")

    property_type = row["property_type"]
    if property_type not in SUPPORTED_PROPERTY_TYPES:
        raise _row_error(
            row_number,
            f"unsupported property_type {property_type!r}; expected one of {sorted(SUPPORTED_PROPERTY_TYPES)}",
        )

    expected_answer = row["expected_answer"]
    if (
        isinstance(expected_answer, bool)
        or not isinstance(expected_answer, (int, float))
        or not math.isfinite(expected_answer)
        or not float(expected_answer).is_integer()
    ):
        raise _row_error(
            row_number,
            f"expected_answer must be integral, got {expected_answer!r}",
        )
    if property_type in BOOLEAN_PROPERTY_TYPES and expected_answer not in {0, 1}:
        raise _row_error(row_number, f"{property_type} expected_answer must be 0 or 1, got {expected_answer}")

    if not isinstance(row["use_box_format"], bool):
        raise _row_error(row_number, "use_box_format must be a boolean")

    responses_create_params = row["responses_create_params"]
    if not isinstance(responses_create_params, Mapping):
        raise _row_error(row_number, "responses_create_params must be an object")

    input_messages = responses_create_params.get("input")
    if not isinstance(input_messages, list) or not input_messages:
        raise _row_error(row_number, "responses_create_params.input must be a non-empty list")
    for message_number, message in enumerate(input_messages, start=1):
        if not isinstance(message, Mapping):
            raise _row_error(row_number, f"input message {message_number} must be an object")
        if not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
            raise _row_error(row_number, f"input message {message_number} must have string role and content")

    if "tools" in responses_create_params:
        tools = responses_create_params["tools"]
        if not isinstance(tools, list):
            raise _row_error(row_number, "responses_create_params.tools must be a list when present")
        if tools:
            raise _row_error(row_number, "v0.1 test rows must not expose tools")

    for field in ("license", "property", "chembl_id", "smiles", "uuid"):
        if not isinstance(row[field], str) or not row[field]:
            raise _row_error(row_number, f"{field} must be a non-empty string")
    if not isinstance(row["messages"], list):
        raise _row_error(row_number, "messages must be a list")
    if not isinstance(row["metadata"], Mapping):
        raise _row_error(row_number, "metadata must be an object")
    if not isinstance(row["used_in"], list) or not all(isinstance(value, str) for value in row["used_in"]):
        raise _row_error(row_number, "used_in must be a list of strings")


def _render_rows(dataset: Any) -> str:
    rendered: list[str] = []
    seen_uuids: set[str] = set()

    for row_number, source_row in enumerate(dataset, start=1):
        if not isinstance(source_row, Mapping):
            raise _row_error(row_number, "row must be an object")

        row = dict(source_row)
        _validate_row(row, row_number)

        # `datasets` infers one schema across the repository's train/test
        # splits, promoting these raw test integers to floats because the train
        # split stores its integral answers as floats. Restore the pinned test
        # file's canonical integer representation.
        row["expected_answer"] = int(row["expected_answer"])

        uuid = row["uuid"]
        if uuid in seen_uuids:
            raise _row_error(row_number, f"duplicate uuid {uuid!r}")
        seen_uuids.add(uuid)

        # The pinned source rows target the removed rdkit_chemistry agent. Gym's
        # benchmark collation assigns the benchmark agent, so do not retain that
        # resource-specific reference in the prepared artifact.
        row.pop("agent_ref", None)
        rendered.append(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    if len(rendered) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Litmus-Bench test split has {len(rendered)} rows; expected {EXPECTED_TEST_ROWS}")

    return "".join(rendered)


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
    """Download, validate, and materialize the pinned v0.1 test split."""
    print(f"Loading {HF_DATASET} ({HF_SPLIT} at {HF_REVISION})...")
    dataset = load_dataset(HF_DATASET, split=HF_SPLIT, revision=HF_REVISION)
    content = _render_rows(dataset)
    _atomic_write(OUTPUT_FPATH, content)
    print(f"Wrote {EXPECTED_TEST_ROWS} Litmus-Bench rows to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
