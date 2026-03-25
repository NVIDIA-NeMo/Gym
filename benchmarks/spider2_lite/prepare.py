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
"""Prepare Spider2 Lite benchmark data."""

from pathlib import Path

from nemo_gym.gitlab_utils import DownloadJsonlDatasetGitlabConfig, download_jsonl_dataset
from resources_servers.spider2_lite.setup_spider2 import ensure_spider2_lite


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "spider2_lite_benchmark.jsonl"


def prepare() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Nvidia-internal only for the moment. Public script coming soon.
    download_jsonl_dataset(
        config=DownloadJsonlDatasetGitlabConfig(
            dataset_name="spider2_lite_sqlite",
            version="0.0.1",
            artifact_fpath="spider2_lite_sqlite_validation.jsonl",
            output_fpath=OUTPUT_FPATH,
        )
    )

    # Download SQL lite databases
    ensure_spider2_lite()

    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
