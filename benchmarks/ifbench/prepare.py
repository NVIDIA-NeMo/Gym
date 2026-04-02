# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Prepare IFBench evaluation data for NeMo Gym.

Downloads IFBench test data from AllenAI's GitHub and converts to Gym JSONL
format compatible with the instruction_following resources server.
"""

import json
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "ifbench_benchmark.jsonl"
RAW_FPATH = DATA_DIR / "IFBench_test.jsonl"
URL = "https://raw.githubusercontent.com/allenai/IFBench/refs/heads/main/data/IFBench_test.jsonl"


def prepare() -> Path:
    """Download IFBench test data and convert to Gym JSONL format."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading IFBench test data from AllenAI GitHub...")
    urllib.request.urlretrieve(URL, RAW_FPATH)

    rows = []
    with open(RAW_FPATH, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            entry = json.loads(line)
            prompt = entry["prompt"]
            rows.append(
                {
                    "id": idx,
                    "instruction_id_list": entry["instruction_id_list"],
                    "prompt": prompt,
                    "kwargs": entry["kwargs"],
                    "responses_create_params": {
                        "input": [{"role": "user", "content": prompt}],
                        "tools": [],
                        "parallel_tool_calls": False,
                    },
                }
            )

    with open(OUTPUT_FPATH, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
