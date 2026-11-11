# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the fixed 100-task WANDR subset: runs the full prepare, which writes both files, and returns the subset.

Gym requires a dataset's prepare() to return that dataset's own jsonl_fpath, so the subset configs use this script.
"""

from pathlib import Path

from benchmarks.wandr import prepare as full


def prepare() -> Path:
    """Download the pinned sources and write the benchmark rows; returns the subset rows' path."""
    full.prepare()
    return full.SUBSET100_OUTPUT_FPATH
