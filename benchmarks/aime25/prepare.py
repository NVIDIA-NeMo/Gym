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

"""Prepare AIME 2025 evaluation data for NeMo Gym.

Downloads AIME 2025 problems from HuggingFace and converts to Gym JSONL format
compatible with the math_with_judge resource server.
"""

import json

from nemo_gym import PARENT_DIR


OUTPUT_PATH = PARENT_DIR / "benchmarks" / "aime25" / "data" / "aime25_validation.jsonl"

SYSTEM_PROMPT = (
    "Your task is to solve a math problem.  Make sure to put the answer (and only the answer) inside \\boxed{}."
)


def prepare():
    """Download AIME 2025 data and convert to Gym JSONL format."""
    from datasets import load_dataset

    print("Downloading AIME 2025 from HuggingFace...")
    ds = load_dataset("MathArena/aime_2025", split="train")

    output_path = OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for example in ds:
        row = {
            "responses_create_params": {
                "input": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ]
            },
            "question": example["problem"],
            "expected_answer": example["answer"],
        }
        rows.append(json.dumps(row) + "\n")

    with open(output_path, "w") as f:
        f.writelines(rows)

    print(f"Wrote {len(rows)} problems to {output_path}")


if __name__ == "__main__":
    prepare()
