# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the official WebVoyager JSONL into normalized Gym rows."""

from __future__ import annotations

import os
from pathlib import Path

from nemo_gym.web.datasets import adapt_webvoyager_record, load_json_records, write_jsonl


BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "webvoyager_benchmark.jsonl"
DEFAULT_SOURCE = BENCHMARK_DIR.parents[2] / "WebVoyager" / "data" / "WebVoyager_data.jsonl"


def prepare(source: str | Path | None = None, output: str | Path = OUTPUT_FPATH) -> Path:
    source_path = Path(source or os.environ.get("WEBVOYAGER_SOURCE_JSONL", DEFAULT_SOURCE))
    rows = [adapt_webvoyager_record(record) for record in load_json_records(source_path)]
    count = write_jsonl(rows, output)
    print(f"Wrote {count} WebVoyager tasks to {output}", flush=True)
    return Path(output)


if __name__ == "__main__":
    prepare()
