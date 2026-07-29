# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare SuperGPQA benchmark data for NeMo Gym."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FPATH = DATA_DIR / "swebench_verified_benchmark.jsonl"


def prepare() -> Path:
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for row in ds:
            row = row | {
                "responses_create_params": {
                    "input": [],
                },
            }
            fout.write(json.dumps(row) + "\n")

    print(f"Wrote {len(ds)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
