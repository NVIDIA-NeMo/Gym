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

"""Prepare AIME 2024 evaluation data for NeMo Gym.

Downloads AIME 2024 problems from HuggingFace and converts to Gym JSONL format
compatible with the math_with_judge resource server.
"""

import argparse
import json
from pathlib import Path

from nemo_gym import PARENT_DIR
from nemo_gym.prompt import load_prompt


BENCHMARK_DIR = PARENT_DIR / "benchmarks" / "aime24"
DEFAULT_PROMPT_CONFIG = str(BENCHMARK_DIR / "prompts" / "default.yaml")


def prepare(prompt_config: str = DEFAULT_PROMPT_CONFIG):
    """Download AIME 2024 data and convert to Gym JSONL format."""
    from datasets import load_dataset

    print("Downloading AIME 2024 from HuggingFace...")
    ds = load_dataset("MathArena/aime_2024", split="train")

    prompt = load_prompt(prompt_config)
    prompt_name = Path(prompt_config).stem
    output_path = BENCHMARK_DIR / "data" / f"aime24_{prompt_name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for example in ds:
        row = {
            "responses_create_params": {
                "input": prompt.fill({"question": example["problem"]}),
            },
            "question": example["problem"],
            "expected_answer": example["answer"],
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
