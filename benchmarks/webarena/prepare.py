# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the official WebArena task configs into Gym benchmark rows."""

from __future__ import annotations

import os
from pathlib import Path

from nemo_gym.web.datasets import adapt_webarena_record, load_json_records, write_jsonl


BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "webarena_benchmark.jsonl"
DEFAULT_SOURCE = BENCHMARK_DIR.parents[2] / "webarena" / "config_files" / "test.raw.json"


def prepare(source: str | Path | None = None, output: str | Path = OUTPUT_FPATH) -> Path:
    source_path = Path(source or os.environ.get("WEBARENA_SOURCE_CONFIG", DEFAULT_SOURCE))
    rows = [adapt_webarena_record(record) for record in load_json_records(source_path)]
    count = write_jsonl(rows, output)
    print(f"Wrote {count} WebArena tasks to {output}", flush=True)
    return Path(output)


if __name__ == "__main__":
    prepare()
