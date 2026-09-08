# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare AA-LCR v1.1 benchmark data."""

from pathlib import Path

from benchmarks.aalcr.prepare import prepare_version


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "aalcr_v1_1_benchmark.jsonl"
DATASET_REVISION = "9a77ef56b717057ade24ceab4d273712a0b4f19e"  # pragma: allowlist secret
BENCHMARK_VERSION = "1.1"
JUDGE_PROTOCOL = "official_v1_1"


def prepare(*, dataset_revision: str = DATASET_REVISION) -> Path:
    if dataset_revision != DATASET_REVISION:
        raise ValueError(f"AA-LCR v1.1 requires dataset revision {DATASET_REVISION}, got {dataset_revision}")
    return prepare_version(
        dataset_revision=dataset_revision,
        benchmark_version=BENCHMARK_VERSION,
        judge_protocol=JUDGE_PROTOCOL,
        output_fpath=OUTPUT_FPATH,
    )


if __name__ == "__main__":
    prepare()
