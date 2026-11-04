# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym preparation entrypoint for the throughput_100k split."""

from pathlib import Path

from .prepare import DATA_DIR, DEFAULT_DATASET_REPO, prepare_one_config


def prepare(
    *,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    dataset_revision: str | None = None,
    output_dir: Path = DATA_DIR,
) -> Path:
    """Prepare throughput_100k using Gym's existing preparation interface."""
    return prepare_one_config(
        "throughput_100k", dataset_repo=dataset_repo, dataset_revision=dataset_revision, output_dir=output_dir
    )
