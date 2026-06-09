# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare SupChain-Bench (SC-bench) data for NeMo Gym.

Downloads upstream SC-bench question/answer JSONL and CSV tables, then converts
them to NeMo Gym JSONL using the ported resources server (tools, evaluator, and
ground-truth orchestration).

Source: https://github.com/Damon-GSY/SC-bench
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


_GYM_ROOT = Path(__file__).resolve().parents[2]
if str(_GYM_ROOT) not in sys.path:
    sys.path.insert(0, str(_GYM_ROOT))

from resources_servers.sc_bench.evaluation import load_ground_truth_lines
from resources_servers.sc_bench.get_results import extract_trade_order_id, get_results
from resources_servers.sc_bench.supchain_tools import SYSTEM_PROMPT, configure_data_dir, to_nemo_gym_tools


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "sc_bench_benchmark.jsonl"
EXAMPLE_FPATH = Path(__file__).resolve().parents[2] / "resources_servers" / "sc_bench" / "data" / "example.jsonl"
TRAIN_FPATH = Path(__file__).resolve().parents[2] / "resources_servers" / "sc_bench" / "data" / "train.jsonl"
VALIDATION_FPATH = Path(__file__).resolve().parents[2] / "resources_servers" / "sc_bench" / "data" / "validation.jsonl"
CSV_DEST_DIR = Path(__file__).resolve().parents[2] / "resources_servers" / "sc_bench" / "data" / "csv"

SC_BENCH_REPO = "https://github.com/Damon-GSY/SC-bench.git"
TRAIN_RATIO = 0.8

_REQUIRED_DATA_FILES = (
    "tool_use_question.jsonl",
    "tool_use_answers.jsonl",
    "TradeOrders.csv",
    "FulfillmentOrders.csv",
    "WarehouseOrders.csv",
    "ErrorLogs.csv",
    "CancellationContext.csv",
)


def _has_sc_bench_data(root: Path) -> bool:
    data_dir = root / "data"
    return all((data_dir / name).exists() for name in _REQUIRED_DATA_FILES)


def _find_local_sc_bench() -> Path | None:
    for path in (BENCHMARK_DIR.parents[1] / "SC-bench", BENCHMARK_DIR.parents[2] / "SC-bench"):
        if _has_sc_bench_data(path):
            return path
    return None


def _ensure_sc_bench_source() -> Path:
    local = _find_local_sc_bench()
    if local is not None:
        return local

    clone_dir = BENCHMARK_DIR / "_sc_bench_src"
    if not _has_sc_bench_data(clone_dir):
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        subprocess.run(
            ["git", "clone", "--depth", "1", SC_BENCH_REPO, str(clone_dir)],
            check=True,
        )
    return clone_dir


def _copy_csv_tables(source_data_dir: Path) -> None:
    CSV_DEST_DIR.mkdir(parents=True, exist_ok=True)
    for name in (
        "TradeOrders.csv",
        "FulfillmentOrders.csv",
        "WarehouseOrders.csv",
        "ErrorLogs.csv",
        "CancellationContext.csv",
    ):
        shutil.copy2(source_data_dir / name, CSV_DEST_DIR / name)


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _gt_lines_for_trade(trade_order_id: str, all_gt_lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [line for line in all_gt_lines if str(line.get("trade_order_id")) == trade_order_id]


def _build_jsonl_row(
    idx: int,
    question: str,
    trade_order_id: str,
    gt_lines: list[dict[str, Any]],
    expected_result: dict[str, Any],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    row_id = trade_order_id or f"idx_{idx}"
    return {
        "id": f"sc_bench_{row_id}_{idx}",
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "tools": tools,
            "parallel_tool_calls": False,
        },
        "verifier_metadata": {
            "trade_order_id": trade_order_id,
            "gt_lines": gt_lines,
            "expected_result": expected_result,
        },
    }


def _load_rows(data_dir: Path) -> list[dict[str, Any]]:
    _, all_gt_lines = load_ground_truth_lines(data_dir / "tool_use_answers.jsonl")
    tools = to_nemo_gym_tools()
    rows: list[dict[str, Any]] = []

    with open(data_dir / "tool_use_question.jsonl", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            record = json.loads(line)
            question = record.get("question", "").strip()
            trade_order_id = extract_trade_order_id(question) or ""
            if not trade_order_id:
                match = re.search(r"T\d+", question.upper())
                trade_order_id = match.group(0) if match else ""

            gt_lines = _gt_lines_for_trade(trade_order_id, all_gt_lines) if trade_order_id else []
            expected_result = get_results(question) if trade_order_id else {}
            rows.append(_build_jsonl_row(idx, question, trade_order_id, gt_lines, expected_result, tools))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_sanitize_for_json(row), ensure_ascii=False) + "\n")


def _convert_to_nemo_gym(
    data_dir: Path,
    output_benchmark: Path,
    output_example: Path,
    output_train: Path,
    output_validation: Path,
) -> int:
    rows = _load_rows(data_dir)
    _write_jsonl(output_benchmark, rows)
    _write_jsonl(output_example, rows[:5])

    split_idx = int(len(rows) * TRAIN_RATIO)
    _write_jsonl(output_train, rows[:split_idx])
    _write_jsonl(output_validation, rows[split_idx:])
    return len(rows)


def prepare() -> Path:
    source_root = _ensure_sc_bench_source()
    source_data = source_root / "data"
    _copy_csv_tables(source_data)
    configure_data_dir(CSV_DEST_DIR)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLE_FPATH.parent.mkdir(parents=True, exist_ok=True)

    count = _convert_to_nemo_gym(
        data_dir=source_data,
        output_benchmark=OUTPUT_FPATH,
        output_example=EXAMPLE_FPATH,
        output_train=TRAIN_FPATH,
        output_validation=VALIDATION_FPATH,
    )

    print(f"Converted {count} tool-use questions.")
    print(f"Wrote benchmark data to {OUTPUT_FPATH}")
    print(f"Wrote example data to {EXAMPLE_FPATH}")
    print(f"Wrote train data to {TRAIN_FPATH}")
    print(f"Wrote validation data to {VALIDATION_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
