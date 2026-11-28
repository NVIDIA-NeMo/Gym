# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download the 2,328-prompt LMArena proxy v4 dataset."""

from pathlib import Path

from nemo_gym.config import DownloadJsonlDatasetGitlabConfig
from nemo_gym.data import download_jsonl_dataset


OUTPUT_FPATH = Path(__file__).parent / "data" / "lmarena_v4_validation.jsonl"


def prepare() -> Path:
    """Download the frozen v4 registry artifact."""
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    download_jsonl_dataset(
        DownloadJsonlDatasetGitlabConfig(
            dataset_name="lmarena_v4",
            version="0.0.1",
            artifact_fpath="lmarena_v4_validation.jsonl",
            output_fpath=str(OUTPUT_FPATH),
        )
    )
    return OUTPUT_FPATH
