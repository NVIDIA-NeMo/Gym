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

"""Prepare CritPt evaluation data for NeMo Gym.

Downloads CritPt-Benchmark/CritPt from HuggingFace (public dataset, no token required)
and converts to Gym JSONL format. Flat-field schema; prompt templating happens at runtime
via the dataset's `prompt_config` pointing at `prompts/turn1.yaml`.
"""

import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "critpt_benchmark.jsonl"

# Note: data/example.jsonl is a hand-curated repo artifact (5 problems spanning multiple
# physics domains) committed under resources_servers/critpt/data/example.jsonl. It is
# intentionally NOT regenerated here — running prepare() will not touch it.


def prepare() -> Path:
    from datasets import load_dataset

    print("Downloading CritPt-Benchmark/CritPt from HuggingFace...")
    ds = load_dataset("CritPt-Benchmark/CritPt", split="train")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for problem in ds:
        row = {
            "problem_id": problem["problem_id"],
            "problem": problem["problem_description"],
            "code_template": problem["code_template"],
            "uuid": problem["problem_id"],
        }
        rows.append(json.dumps(row) + "\n")

    with open(OUTPUT_FPATH, "w") as f:
        f.writelines(rows)
    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")

    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
