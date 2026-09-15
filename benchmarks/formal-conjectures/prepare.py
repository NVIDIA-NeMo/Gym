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

"""Prepare the Formal Conjectures benchmark for NeMo Gym."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from resources_servers.formal_conjectures.prepare import build_rows, write_jsonl  # noqa: E402


OUTPUT_FPATH = Path(__file__).parent / "data" / "formal_conjectures_benchmark.jsonl"


def prepare() -> Path:
    """Fetch the pinned Formal Conjectures revision and write the benchmark JSONL."""
    write_jsonl(OUTPUT_FPATH, build_rows())
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
