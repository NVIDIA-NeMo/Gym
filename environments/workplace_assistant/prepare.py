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
"""Prepare workplace assistant training data.

Downloads train and validation splits from HuggingFace and converts them
to Gym JSONL format.

Usage:
    python environments/workplace_assistant/prepare.py
    python environments/workplace_assistant/prepare.py --split train
    python environments/workplace_assistant/prepare.py --split validation
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from resources_servers.workplace_assistant.utils import get_tools


ENV_DIR = Path(__file__).parent
DATA_DIR = ENV_DIR / "data"

HF_REPO_ID = "Nexusflow/250319-workplace_assistant-fulleval"

HARDCODED_CURRENT_TIME = pd.to_datetime("2023-11-30T23:59:00")

SYS_PROMPT = (
    f"Today's date is {HARDCODED_CURRENT_TIME.strftime('%A')}, {HARDCODED_CURRENT_TIME.date()} "
    f"and the current time is {HARDCODED_CURRENT_TIME.time()}. Remember the current date and time when answering queries. "
    "Meetings must not start before 9am or end after 6pm."
)

TOOLKITS = ["email", "calendar", "analytics", "project_management", "customer_relationship_manager"]

# HuggingFace split names
_HF_SPLITS = {"train": "train", "validation": "test"}


def prepare(split: str = "train") -> Path:
    """Download and prepare data for the given split. Returns the output file path."""
    if split not in _HF_SPLITS:
        raise ValueError(f"split must be one of {list(_HF_SPLITS)}, got '{split}'")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_fpath = DATA_DIR / f"{split}.jsonl"

    tool_env = get_tools(TOOLKITS)
    base_create_params = dict(
        input=[{"role": "system", "content": SYS_PROMPT}],
        tools=tool_env.get("schemas"),
        parallel_tool_calls=False,
        temperature=1.0,
    )

    hf_split = _HF_SPLITS[split]
    print(f"Loading workplace_assistant {split} data from {HF_REPO_ID} (split={hf_split})...")
    ds = load_dataset(HF_REPO_ID, split=hf_split)

    count = 0
    with open(output_fpath, "w") as f:
        for i, row in enumerate(ds):
            create_params = deepcopy(base_create_params)
            create_params["input"].append({"role": "user", "content": row["problem"]})

            record = {
                "id": i,
                "responses_create_params": create_params,
                "verifier_metadata": {
                    "ground_truth": json.loads(row["solution"]),
                    "category": row["category"],
                    "environment_name": row["environment_name"],
                },
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    print(f"Wrote {count} samples to {output_fpath}")
    return output_fpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=list(_HF_SPLITS),
        default=None,
        help="Which split to prepare. Defaults to both train and validation.",
    )
    args = parser.parse_args()

    splits = [args.split] if args.split else list(_HF_SPLITS)
    for s in splits:
        prepare(s)
