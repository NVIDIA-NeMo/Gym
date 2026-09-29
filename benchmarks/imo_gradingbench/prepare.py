# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare IMO-GradingBench evaluation data for NeMo Gym.

Direct port of ``nemo_skills/dataset/imo-gradingbench/prepare.py``.
Downloads ``gradingbench.csv`` from the same pinned commit
(``c1ee02e0…``) of ``google-deepmind/superhuman/imobench`` and writes
one JSONL row per ``Grading ID``. Field names mirror Skills exactly so
the prompt template (``{problem_statement}``, ``{proof}``) and the
verifier's ``expected_answer`` field stay in sync.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "imo_gradingbench_benchmark.jsonl"

BASE_URL = (
    "https://raw.githubusercontent.com/google-deepmind/superhuman/c1ee02e03d4cdb2ab21cd01ac927d895f5287fc8/imobench"
)
SOURCE_URL = f"{BASE_URL}/gradingbench.csv"


def prepare() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading IMO-GradingBench CSV from {SOURCE_URL}...")
    with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
        content = response.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(content))

    n = 0
    with OUTPUT_FPATH.open("w", encoding="utf-8") as out:
        for row in reader:
            entry = {
                "grading_id": row["Grading ID"],
                "problem_id": row["Problem ID"],
                "problem_statement": row["Problem"],
                "reference_solution": row["Solution"],
                "rubric": row["Grading guidelines"],
                "proof": row["Response"],
                "points": row["Points"],
                "reward": row["Reward"],
                "source": row["Problem Source"],
                # Verifier reads `expected_answer` directly. Skills' prepare.py
                # also sets this to the gold reward word so the four-class
                # extractor can match against it.
                "expected_answer": row["Reward"],
            }
            out.write(json.dumps(entry) + "\n")
            n += 1

    print(f"Wrote {n} rows to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
