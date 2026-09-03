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
"""Prepare LMArena proxy v2 benchmark data.

Downloads the v2 dataset from the NVIDIA GitLab Model Registry.
"""

from pathlib import Path

from nemo_gym.config_types import DownloadJsonlDatasetGitlabConfig
from nemo_gym.gitlab_utils import download_jsonl_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "lmarena_v2_validation.jsonl"


def prepare() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_jsonl_dataset(
        DownloadJsonlDatasetGitlabConfig(
            dataset_name="lmarena_v2",
            version="0.0.1",
            artifact_fpath="lmarena_v2_validation.jsonl",
            output_fpath=str(OUTPUT_FPATH),
        )
    )
    return OUTPUT_FPATH
