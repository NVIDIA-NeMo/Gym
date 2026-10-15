# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
from pathlib import Path


DATASET_REPO = "JobBench/job-bench"
DATASET_REVISION = "d43a66575635da0d3f6cb9a20976dc1a60f3ffc9"  # pragma: allowlist secret


@lru_cache
def snapshot_root(source_dir: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            DATASET_REPO,
            repo_type="dataset",
            revision=DATASET_REVISION,
            allow_patterns=f"{source_dir}/**",
        )
    )


def resolve_task_path(path: str) -> Path:
    # Dataset rows store paths relative to the pinned snapshot, so they resolve on any machine.
    task_path = Path(path)
    return task_path if task_path.is_absolute() else snapshot_root(task_path.parts[0]) / task_path
