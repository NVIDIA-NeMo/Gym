# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Math-500 benchmark data for NeMo Gym."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "math-500_benchmark.jsonl"
URL = "https://github.com/openai/prm800k/raw/main/prm800k/math_splits/test.jsonl"


def prepare() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    original_fpath = DATA_DIR / "original_test.jsonl"

    print(f"Downloading Math-500 from {URL}")
    urllib.request.urlretrieve(URL, original_fpath)

    count = 0
    with original_fpath.open("r", encoding="utf-8") as fin, OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            entry = json.loads(line)
            problem = entry.get("problem", entry.get("question"))
            if problem is None:
                raise ValueError("Expected `problem` or `question` in Math-500 row")

            row = dict(entry)
            row["question"] = problem
            row["problem"] = problem
            row["expected_answer"] = row.pop("answer")
            row["reference_solution"] = row.pop("solution")
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    original_fpath.unlink()
    print(f"Wrote {count} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
