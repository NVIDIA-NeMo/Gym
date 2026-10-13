# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare BunsenBench Chemistry MCQ data from the upstream Hugging Face dataset."""

from __future__ import annotations

from pathlib import Path

from benchmarks.bunsenbench_chemistry_mcq.materialize import materialize_dataset
from benchmarks.bunsenbench_chemistry_mcq.upstream import reconstitute_upstream_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "bunsenbench_chemistry_mcq_benchmark.jsonl"


def prepare(output_path: Path = OUTPUT_FPATH, *, limit: int | None = None) -> Path:
    """Reconstitute upstream BunsenBench Chemistry MCQ rows and materialize Gym JSONL."""
    dataset = reconstitute_upstream_dataset(limit=limit)
    return materialize_dataset(dataset, output_path)


if __name__ == "__main__":
    prepare()
