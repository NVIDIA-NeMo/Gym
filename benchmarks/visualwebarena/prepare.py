# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert official VWA partitions using BrowserGym's global task-id order."""

from __future__ import annotations

import os
from pathlib import Path

from nemo_gym.web.datasets import adapt_visualwebarena_records, load_json_records, write_jsonl


BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "visualwebarena_benchmark.jsonl"
DEFAULT_SOURCE_DIR = BENCHMARK_DIR.parents[2] / "visualwebarena" / "config_files" / "vwa"
PARTITIONS = (
    ("classifieds", "test_classifieds.raw.json"),
    ("reddit", "test_reddit.raw.json"),
    ("shopping", "test_shopping.raw.json"),
)


def prepare(source_dir: str | Path | None = None, output: str | Path = OUTPUT_FPATH) -> Path:
    root = Path(source_dir or os.environ.get("VISUALWEBARENA_SOURCE_DIR", DEFAULT_SOURCE_DIR))
    partitions = [(name, load_json_records(root / filename)) for name, filename in PARTITIONS]
    rows = adapt_visualwebarena_records(partitions)
    if len(rows) != 910:
        raise ValueError(f"expected 910 VisualWebArena tasks, found {len(rows)}")
    count = write_jsonl(rows, output)
    print(f"Wrote {count} VisualWebArena tasks to {output}", flush=True)
    return Path(output)


if __name__ == "__main__":
    prepare()
