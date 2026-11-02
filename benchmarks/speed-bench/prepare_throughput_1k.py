# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym preparation entrypoint for the throughput_1k split."""

import sys
from pathlib import Path


# Gym imports this file by module path; make the shared sibling helper available.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _prepare_common import DEFAULT_DATASET_REPO, prepare_one_config  # noqa: E402


def prepare(
    *,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    dataset_revision: str | None = None,
) -> Path:
    """Prepare throughput_1k using Gym's existing preparation interface."""
    return prepare_one_config("throughput_1k", dataset_repo=dataset_repo, dataset_revision=dataset_revision)
