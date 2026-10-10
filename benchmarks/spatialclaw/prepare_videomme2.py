# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the Video-MME-v2 SpatialClaw benchmark."""

from pathlib import Path

from benchmarks.spatialclaw.prepare import prepare as prepare_dataset


OUTPUT_FPATH = Path(__file__).parent / "data" / "videomme2_benchmark.jsonl"


def prepare(**kwargs) -> Path:
    return prepare_dataset(
        dataset_config="videommev2_spatialclaw_256f",
        output_fpath=OUTPUT_FPATH,
        **kwargs,
    )


if __name__ == "__main__":
    prepare()
