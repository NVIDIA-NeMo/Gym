# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the audiobench.nonjudge BLEU bucket — 6 covost2 translation pairs."""

from __future__ import annotations

from pathlib import Path

from benchmarks.audiobench._prepare_lib import main_for_bucket, prepare_audiobench_bucket


def prepare() -> Path:
    return prepare_audiobench_bucket(bucket="bleu")


if __name__ == "__main__":
    main_for_bucket("bleu")
