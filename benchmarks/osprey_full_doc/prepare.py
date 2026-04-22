# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "osprey_full_doc_benchmark.jsonl"
SOURCE_SNAPSHOT_FPATH = DATA_DIR / "osprey_extraction_benchmark_dataset_full_doc.json"
PINNED_TEMPERATURE = 1.0
PINNED_TOP_P = 0.95
PINNED_MAX_OUTPUT_TOKENS = 32000


def _resolve_source_fpath(source_json: str | None = None) -> Path:
    candidate = source_json or os.environ.get("OSPREY_FULL_DOC_SOURCE_JSON")
    if candidate:
        path = Path(candidate).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"Osprey full_doc source data file is missing: {path}")

    raise FileNotFoundError(
        "Could not find Osprey full_doc source data. Pass `source_json=...`, use "
        "`--source-json`, or set OSPREY_FULL_DOC_SOURCE_JSON before running "
        "ng_prepare_benchmark."
    )


def _convert_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool["function"]
    return {
        "type": "function",
        "name": function["name"],
        "description": function.get("description", ""),
        "parameters": function.get("parameters", {}),
        "strict": False,
    }


def _convert_row(row: dict[str, Any]) -> dict[str, Any]:
    doc_name = row["doc_name"]
    line_item_name = row["line_item_name"]
    return {
        "id": f"{doc_name}::{line_item_name}",
        "doc_name": doc_name,
        "line_item_name": line_item_name,
        "ground_truth": row.get("ground_truth"),
        "responses_create_params": {
            "input": row["messages"],
            "tools": [_convert_tool(tool) for tool in row["tools"]],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "temperature": PINNED_TEMPERATURE,
            "top_p": PINNED_TOP_P,
            "max_output_tokens": PINNED_MAX_OUTPUT_TOKENS,
        },
    }


def _snapshot_source_json(source_fpath: Path) -> Path:
    if source_fpath.resolve() == SOURCE_SNAPSHOT_FPATH.resolve():
        return SOURCE_SNAPSHOT_FPATH

    shutil.copy2(source_fpath, SOURCE_SNAPSHOT_FPATH)
    return SOURCE_SNAPSHOT_FPATH


def prepare(source_json: str | None = None, copy_source_to_data_dir: bool = True) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    source_fpath = _resolve_source_fpath(source_json=source_json)
    with source_fpath.open("rt", encoding="utf-8") as f:
        source_rows = json.load(f)

    with OUTPUT_FPATH.open("wt", encoding="utf-8") as f:
        for row in source_rows:
            f.write(json.dumps(_convert_row(row)) + "\n")

    if copy_source_to_data_dir:
        snapshot_fpath = _snapshot_source_json(source_fpath)
        print(f"Snapshotted source JSON to {snapshot_fpath}")

    print(f"Wrote {len(source_rows)} rows to {OUTPUT_FPATH} from {source_fpath}")
    return OUTPUT_FPATH


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Osprey full_doc benchmark JSONL for NeMo Gym.")
    parser.add_argument(
        "--source-json",
        default=None,
        help="Path to the original Osprey full_doc source JSON.",
    )
    parser.add_argument(
        "--skip-copy-source",
        action="store_true",
        help="Do not copy the original source JSON into benchmarks/osprey_full_doc/data/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare(
        source_json=args.source_json,
        copy_source_to_data_dir=not args.skip_copy_source,
    )
