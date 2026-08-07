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
"""Prepare the AgentIF gym-native benchmark dataset.

The resources server ships ``prepare_agentif.py``, which converts the upstream
``THU-KEG/AgentIF`` eval.json into gym rows (Responses-shaped ``input`` plus the
gold ``verifier_metadata`` the judge/code checkers score against). This wrapper
reuses its ``build_row`` to write the whole 707-row dataset to
``data/agentif_benchmark.jsonl`` for the gym-native eval.

Usage::

    python benchmarks/agentif/prepare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


_RESOURCES_SERVER_DIR = Path(__file__).resolve().parents[2] / "resources_servers" / "agentif"
if str(_RESOURCES_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_RESOURCES_SERVER_DIR))

from prepare_agentif import _DEFAULT_INPUT, build_row  # noqa: E402


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "agentif_benchmark.jsonl"


def prepare(input_fpath: Path = _DEFAULT_INPUT) -> Path:
    """Build the whole-dataset AgentIF benchmark JSONL."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(input_fpath, encoding="utf-8") as reader:
        data = json.load(reader)
    with open(OUTPUT_FPATH, "w", encoding="utf-8") as writer:
        for row in data:
            writer.write(json.dumps(build_row(row), ensure_ascii=False) + "\n")
    print(f"AgentIF: wrote {len(data)} rows -> {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
