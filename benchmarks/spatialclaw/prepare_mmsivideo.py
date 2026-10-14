# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the mmsivideo SpatialClaw preset."""

from pathlib import Path

from benchmarks.spatialclaw.prepare import prepare as prepare_dataset


OUTPUT_FPATH = Path(__file__).parent / "data" / "mmsivideo_benchmark.jsonl"


def prepare(**kwargs) -> Path:
    return prepare_dataset(
        dataset_config="mmsivideo_spatialclaw_256f",
        output_fpath=OUTPUT_FPATH,
        **kwargs,
    )


if __name__ == "__main__":
    prepare()
