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

"""Prepare SciCode evaluation data for NeMo Gym.

Downloads the SciCode problems from HuggingFace and converts to Gym JSONL, one row per problem.
Each row carries the full sub_steps list so the agent can build the per-sub-step prompts and the
resources server can run each sub-step's test cases.

The benchmark split is validation + test combined (what nemo-skills calls its "test_aai" split).

This does NOT download test_data.h5 (the ~1 GB numeric test targets used for scoring). That file
must be staged manually at the path the resources server is configured to read.
"""

from pathlib import Path

from nemo_gym.global_config import HF_TOKEN_KEY_NAME, get_global_config_dict


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "scicode_benchmark.jsonl"

# Evaluate on all 80 problems
SPLITS = ["validation", "test"]


def prepare() -> Path:
    """Download SciCode problems and convert to Gym JSONL format."""
    import json

    from datasets import load_dataset

    print("Downloading SciCode from HuggingFace...")
    hf_token = get_global_config_dict().get(HF_TOKEN_KEY_NAME)
    ds = load_dataset("SciCode1/SciCode", token=hf_token)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in SPLITS:
        for entry in ds[split]:
            row = {
                # The agent builds each sub-step's messages from sub_steps, so the row carries an
                # empty input; rollout collection still requires the key to be present.
                "responses_create_params": {"input": []},
                "problem_id": entry["problem_id"],
                "problem_name": entry["problem_name"],
                "required_dependencies": entry["required_dependencies"],
                "sub_steps": entry["sub_steps"],
                # problem_id is unique across both splits, so it doubles as the rollout uuid.
                "uuid": entry["problem_id"],
            }
            rows.append(json.dumps(row) + "\n")

    with open(OUTPUT_FPATH, "w") as f:
        f.writelines(rows)

    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
