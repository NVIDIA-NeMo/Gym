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

"""Prepare GPQA Diamond evaluation data for NeMo Gym.

Downloads GPQA Diamond from HuggingFace and converts to Gym JSONL format
compatible with the mcqa resource server.
"""

import argparse
import hashlib
import json
import random
import uuid
from pathlib import Path

from nemo_gym import PARENT_DIR
from nemo_gym.prompt import load_prompt


BENCHMARK_DIR = PARENT_DIR / "benchmarks" / "gpqa"
DEFAULT_PROMPT_CONFIG = str(BENCHMARK_DIR / "prompts" / "default.yaml")
OPTION_LETTERS = ["A", "B", "C", "D"]


def prepare(prompt_config: str = DEFAULT_PROMPT_CONFIG):
    """Download GPQA Diamond data and convert to Gym JSONL format."""
    from datasets import load_dataset

    print("Downloading GPQA Diamond from HuggingFace...")
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")

    prompt = load_prompt(prompt_config)
    prompt_name = Path(prompt_config).stem
    output_path = BENCHMARK_DIR / "data" / f"gpqa_diamond_{prompt_name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for example in ds:
        # Build options list from the dataset columns
        choices = [
            example["Correct Answer"],
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]

        # Shuffle options deterministically using the question as seed
        seed = int(hashlib.md5(example["Question"].encode()).hexdigest(), 16)
        rng = random.Random(seed)
        rng.shuffle(choices)

        # Find which letter is the correct answer after shuffle
        correct_idx = choices.index(example["Correct Answer"])
        correct_letter = OPTION_LETTERS[correct_idx]

        # Format options as MCQA expects
        options = [{letter: text} for letter, text in zip(OPTION_LETTERS, choices)]
        options_text = "\n".join(f"{letter}: {text}" for letter, text in zip(OPTION_LETTERS, choices))

        row = {
            "responses_create_params": {
                "input": prompt.fill({"question": example["Question"], "options_text": options_text}),
            },
            "options": options,
            "expected_answer": correct_letter,
            "grading_mode": "strict_single_letter_boxed",
            "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, example["Question"])),
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
