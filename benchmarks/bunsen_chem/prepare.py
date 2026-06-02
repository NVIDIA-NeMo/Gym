# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare BunsenChem benchmark data from the upstream Hugging Face dataset."""

from __future__ import annotations

from pathlib import Path

from benchmarks.bunsen_chem.materialize import materialize_dataset
from benchmarks.bunsen_chem.upstream import reconstitute_upstream_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "bunsen_chem_benchmark.jsonl"


def prepare(output_path: Path = OUTPUT_FPATH, *, limit: int | None = None) -> Path:
    """Reconstitute upstream BunsenChem rows and materialize Gym JSONL."""
    dataset = reconstitute_upstream_dataset(limit=limit)
    return materialize_dataset(dataset, output_path)


if __name__ == "__main__":
    prepare()
