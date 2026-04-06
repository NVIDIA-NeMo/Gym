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

from argparse import Namespace
from pathlib import Path
from shutil import copy

from nemo_gym.resources_servers.spider2_lite.scripts.prepare_dataset import (
    _main,
    clone_spider2_repo,
    delete_spider2_repo,
)
from nemo_gym.resources_servers.spider2_lite.setup_spider2 import _DEFAULT_DIR, ensure_spider2_lite


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "spider2_lite_benchmark.jsonl"


def prepare() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download SQL lite databases
    ensure_spider2_lite()

    clone_spider2_repo(parent_dir=_DEFAULT_DIR)

    _main(
        args=Namespace(
            spider2_dir=_DEFAULT_DIR / "Spider2" / "spider2-lite",
            sqlite_dir=None,
            output_dir=str(OUTPUT_FPATH.parent),
        )
    )
    copy(
        OUTPUT_FPATH.parent / "spider2_lite_sqlite_validation.jsonl",
        OUTPUT_FPATH,
    )

    delete_spider2_repo(parent_dir=_DEFAULT_DIR)

    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
