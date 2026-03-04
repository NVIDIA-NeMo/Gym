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

"""Prepare LiveCodeBench evaluation data for NeMo Gym.

Downloads LiveCodeBench v5 from HuggingFace and converts to Gym JSONL format
compatible with the code_gen resource server.
"""

import json

from nemo_gym import PARENT_DIR


OUTPUT_PATH = (
    PARENT_DIR / "resources_servers" / "code_gen" / "data" / "livecodebench_v5_2024-07-01_2025-02-01_validation.jsonl"
)

SYSTEM_PROMPT = (
    "You are a helpful and harmless assistant. You should think step-by-step before responding to the instruction"
    " below.\n\nPlease use python programming language only.\n\nYou must use ```python for just the final solution"
    " code block with the following format:\n```python\n# Your code here\n```"
)

# LiveCodeBench date range for v5
DATE_FROM = "2024-07-01"
DATE_TO = "2025-02-01"


def prepare():
    """Download LiveCodeBench data and convert to Gym JSONL format."""
    from datasets import load_dataset

    print("Downloading LiveCodeBench from HuggingFace...")
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        "release_v5",
        split="test",
        revision="refs/pr/7",
    )

    output_path = OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for example in ds:
        # Filter by date range for v5
        contest_date = example.get("contest_date", "")
        if contest_date and (contest_date < DATE_FROM or contest_date >= DATE_TO):
            continue

        # Build unit tests from public test cases
        input_output = (
            json.loads(example["public_test_cases"])
            if isinstance(example["public_test_cases"], str)
            else example["public_test_cases"]
        )
        inputs = [tc["input"] for tc in input_output]
        outputs = [tc["output"] for tc in input_output]

        # Also include hidden test cases if available
        if example.get("hidden_test_cases"):
            hidden = (
                json.loads(example["hidden_test_cases"])
                if isinstance(example["hidden_test_cases"], str)
                else example["hidden_test_cases"]
            )
            inputs.extend(tc["input"] for tc in hidden)
            outputs.extend(tc["output"] for tc in hidden)

        row = {
            "responses_create_params": {
                "input": [
                    {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{example['question_content']}"},
                ]
            },
            "verifier_metadata": {
                "unit_tests": {
                    "inputs": inputs,
                    "outputs": outputs,
                    "fn_name": None,
                }
            },
            "problem_id": example.get("question_id", ""),
        }
        rows.append(json.dumps(row) + "\n")

    with open(output_path, "w") as f:
        f.writelines(rows)

    print(f"Wrote {len(rows)} problems to {output_path}")


if __name__ == "__main__":
    prepare()
