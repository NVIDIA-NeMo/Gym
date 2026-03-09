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

import argparse
import json
from pathlib import Path

from nemo_gym import PARENT_DIR
from nemo_gym.prompt import load_prompt


BENCHMARK_DIR = PARENT_DIR / "benchmarks" / "livecodebench"
DEFAULT_PROMPT_CONFIG = str(BENCHMARK_DIR / "prompts" / "default.yaml")

# LiveCodeBench date range for v5
DATE_FROM = "2024-07-01"
DATE_TO = "2025-02-01"


def prepare(prompt_config: str = DEFAULT_PROMPT_CONFIG):
    """Download LiveCodeBench data and convert to Gym JSONL format."""
    from datasets import load_dataset

    print("Downloading LiveCodeBench from HuggingFace...")
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        "release_v5",
        split="test",
        revision="refs/pr/7",
    )

    prompt = load_prompt(prompt_config)
    prompt_name = Path(prompt_config).stem
    output_path = BENCHMARK_DIR / "data" / f"livecodebench_v5_{prompt_name}.jsonl"
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
                "input": prompt.fill({"question_content": example["question_content"]}),
            },
            "verifier_metadata": {
                "unit_tests": {
                    "inputs": inputs,
                    "outputs": outputs,
                    "fn_name": None,
                }
            },
            "problem_id": example.get("question_id", ""),
            "prompt_config_used": prompt_config,
        }
        rows.append(json.dumps(row) + "\n")

    with open(output_path, "w") as f:
        f.writelines(rows)

    print(f"Wrote {len(rows)} problems to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_config", default=DEFAULT_PROMPT_CONFIG)
    args = parser.parse_args()
    prepare(prompt_config=args.prompt_config)
