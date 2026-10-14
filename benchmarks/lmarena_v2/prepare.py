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

Downloads the v2 dataset and applies generation defaults to its rows.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from nemo_gym.config_types import DownloadJsonlDatasetGitlabConfig
from nemo_gym.gitlab_utils import download_jsonl_dataset


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "lmarena_v2_validation.jsonl"
GENERATION_DEFAULTS = {"temperature": 1.0, "top_p": 0.95, "max_output_tokens": 16384, "stream": False}


def is_prepared_data_current(fpath: Path) -> bool:
    """Whether a cached prepared file has rows and carries the generation defaults on every one."""
    has_rows = False
    try:
        with fpath.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                has_rows = True
                params = json.loads(line).get("responses_create_params") or {}
                if any(params.get(key) != value for key, value in GENERATION_DEFAULTS.items()):
                    return False
    except (json.JSONDecodeError, AttributeError):
        return False
    return has_rows


def prepare() -> Path:
    """Prepare benchmark rows with generation settings scoped to LMArena."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Row-local defaults preserve the standalone recipe without overriding other benchmarks.
    # Stage both files so a failed download or transformation keeps the previous output.
    with TemporaryDirectory(dir=DATA_DIR) as temporary_dir:
        source_fpath = Path(temporary_dir) / "source.jsonl"
        download_jsonl_dataset(
            DownloadJsonlDatasetGitlabConfig(
                dataset_name="lmarena_v2",
                version="0.0.1",
                artifact_fpath="lmarena_v2_validation.jsonl",
                output_fpath=str(source_fpath),
            )
        )
        staged_fpath = Path(temporary_dir) / OUTPUT_FPATH.name
        with source_fpath.open() as source, staged_fpath.open("w") as target:
            for line in source:
                row = json.loads(line)
                row["responses_create_params"].update(GENERATION_DEFAULTS)
                target.write(json.dumps(row) + "\n")
        staged_fpath.replace(OUTPUT_FPATH)
    return OUTPUT_FPATH
