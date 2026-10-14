# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare audiobench/exact_match — 4 Spoken-MQA math sub-datasets unified into one JSONL."""

from __future__ import annotations

from pathlib import Path

from benchmarks.audiobench._prepare_lib import main_for_bucket, prepare_audiobench_bucket


def prepare(
    *,
    datasets: list[str] | None = None,
    max_samples_per_dataset: int | None = None,
    audio_prefix: str | None = None,
) -> Path:
    """Entry point invoked by ``gym eval prepare --benchmark audiobench/exact_match``."""
    return prepare_audiobench_bucket(
        "exact_match",
        datasets=datasets,
        max_samples_per_dataset=max_samples_per_dataset,
        audio_prefix=audio_prefix,
    )


if __name__ == "__main__":
    main_for_bucket("exact_match")
