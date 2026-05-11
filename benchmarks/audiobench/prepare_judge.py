# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the audiobench.judge bucket — 32 sub-datasets unified into one JSONL."""

from __future__ import annotations

from pathlib import Path

from benchmarks.audiobench._prepare_lib import main_for_bucket, prepare_audiobench_bucket


def prepare() -> Path:
    """Entry point invoked by ng_prepare_benchmark."""
    return prepare_audiobench_bucket(bucket="judge")


if __name__ == "__main__":
    main_for_bucket("judge")
