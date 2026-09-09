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
"""Prepare LeanCat (upstream repo prompt) for the `leancat` resources server."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from resources_servers.leancat.prepare_leancat import build_rows, write_jsonl  # noqa: E402


OUTPUT_FPATH = Path(__file__).parent / "data" / "leancat_benchmark.jsonl"
PROMPT_VARIANT = "repo"


def prepare() -> Path:
    """Fetch the pinned LeanCat revision and write the benchmark JSONL. Returns its path."""
    rows = build_rows(PROMPT_VARIANT)
    write_jsonl(OUTPUT_FPATH, rows)
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
