# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unified SPEED-Bench preparation entrypoint for all six public configurations."""

import sys
from pathlib import Path


# The benchmark dir name has a hyphen, so importlib's dotted-path import
# (`benchmarks.speed-bench.prepare`) is not a valid identifier and the
# parent dir isn't auto-added to sys.path. Prepend it so the sibling
# `_prepare_common` module is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _prepare_common import DEFAULT_DATASET_REPO, prepare_cli, prepare_one_config  # noqa: E402


def prepare(
    speed_config: str = "qualitative",
    *,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    dataset_revision: str | None = None,
) -> Path:
    """Prepare a split from the selected HF repository and return its Gym JSONL path."""
    return prepare_one_config(speed_config, dataset_repo=dataset_repo, dataset_revision=dataset_revision)


if __name__ == "__main__":
    prepare_cli()
