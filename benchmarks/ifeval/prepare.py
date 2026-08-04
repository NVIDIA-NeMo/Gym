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

"""Prepare IFEval evaluation data for NeMo Gym.

Downloads the original IFEval test data from the google-research GitHub repo
and converts it to Gym JSONL format compatible with the
`instruction_following` resources server.

Each output row has two keys:
  responses_create_params  — the agent input (user message with the prompt)
  verifier_metadata        — scoring fields read by the resources server

grading_mode is set to "binary" to align with the original paper's
prompt-level strict accuracy metric (1 if all instructions pass, 0 otherwise).
"""

import json
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
RAW_FPATH = DATA_DIR / "input_data.jsonl"
OUTPUT_FPATH = DATA_DIR / "ifeval_benchmark.jsonl"
URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "master/instruction_following_eval/data/input_data.jsonl"
)


def prepare() -> Path:
    """Download IFEval input data and convert to Gym JSONL format."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading IFEval input data from {URL} ...")
    urllib.request.urlretrieve(URL, RAW_FPATH)

    rows = []
    with open(RAW_FPATH, "rt", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            entry = json.loads(line)
            prompt = entry["prompt"]
            row = {
                "id": idx,
                "responses_create_params": {
                    "input": [{"role": "user", "content": prompt}],
                    "tools": [],
                    "parallel_tool_calls": False,
                },
                "verifier_metadata": {
                    "prompt": prompt,
                    "instruction_id_list": entry["instruction_id_list"],
                    "kwargs": entry["kwargs"],
                    "grading_mode": "binary",
                },
            }
            rows.append(row)

    with open(OUTPUT_FPATH, "wt", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
