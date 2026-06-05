# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare SupChain-Bench (SC-bench) data for NeMo Gym.

Delegates JSONL conversion to the upstream SC-bench repository script:
  SC-bench/scripts/convert_to_nemo_gym.py

Source: https://github.com/Damon-GSY/SC-bench
"""

import shutil
import subprocess
import sys
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "sc_bench_benchmark.jsonl"
EXAMPLE_FPATH = Path(__file__).resolve().parents[2] / "resources_servers" / "sc_bench" / "data" / "example.jsonl"
TRAIN_FPATH = Path(__file__).resolve().parents[2] / "resources_servers" / "sc_bench" / "data" / "train.jsonl"
VALIDATION_FPATH = Path(__file__).resolve().parents[2] / "resources_servers" / "sc_bench" / "data" / "validation.jsonl"
CSV_DEST_DIR = Path(__file__).resolve().parents[2] / "resources_servers" / "sc_bench" / "data" / "csv"

SC_BENCH_REPO = "https://github.com/aneesh-iyer29/SC-bench"


def _find_local_sc_bench() -> Path | None:
    for path in (BENCHMARK_DIR.parents[1] / "SC-bench", BENCHMARK_DIR.parents[2] / "SC-bench"):
        if (path / "scripts" / "convert_to_nemo_gym.py").exists():
            return path
    return None


def _ensure_sc_bench_source() -> Path:
    local = _find_local_sc_bench()
    if local is not None:
        return local

    clone_dir = BENCHMARK_DIR / "_sc_bench_src"
    if not (clone_dir / "scripts" / "convert_to_nemo_gym.py").exists():
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


def prepare() -> Path:
    source_root = _ensure_sc_bench_source()
    source_data = source_root / "data"
    _copy_csv_tables(source_data)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLE_FPATH.parent.mkdir(parents=True, exist_ok=True)

    convert_script = source_root / "scripts" / "convert_to_nemo_gym.py"
    subprocess.run(
        [
            sys.executable,
            str(convert_script),
            "--data-dir",
            str(source_data),
            "--output-benchmark",
            str(OUTPUT_FPATH),
            "--output-example",
            str(EXAMPLE_FPATH),
            "--output-train",
            str(TRAIN_FPATH),
            "--output-validation",
            str(VALIDATION_FPATH),
        ],
        check=True,
        cwd=source_root,
    )

    print(f"Wrote benchmark data to {OUTPUT_FPATH}")
    print(f"Wrote example data to {EXAMPLE_FPATH}")
    print(f"Wrote train data to {TRAIN_FPATH}")
    print(f"Wrote validation data to {VALIDATION_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
