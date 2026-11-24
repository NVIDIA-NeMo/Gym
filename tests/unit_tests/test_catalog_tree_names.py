# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A top-level folder name in the repository is used in only one of benchmarks/ and environments/."""

import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def _tracked_top_level_folders(tree: str) -> set[str]:
    # Tracked files only: a checkout can keep an ignored data folder under a name a rename moved away from.
    tracked = subprocess.run(
        ["git", "ls-files", "--", f"{tree}/"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return {Path(path).parts[1] for path in tracked if len(Path(path).parts) > 2}


def test_no_top_level_folder_name_is_both_a_benchmark_and_an_environment() -> None:
    # A catalog name is its folder path and identifies one entry whatever its kind, as
    # nemo_gym.environment.registry.folder_in_other_catalog_tree describes.
    benchmarks = _tracked_top_level_folders("benchmarks")
    environments = _tracked_top_level_folders("environments")
    assert benchmarks and environments

    assert sorted(benchmarks & environments) == []
